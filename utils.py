import anthropic
import os
import sys
import time
import traceback
from dotenv import load_dotenv

load_dotenv()

DEBUG = "--debug" in sys.argv or os.environ.get("DEBUG_STREAM") == "1"

client = anthropic.AsyncAnthropic()

def add_user_messages(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

async def chat(messages, system_prompt=None, effort=None, stream=False):
    kwargs = {
        "model": os.environ["ANTHROPIC_MODEL"],
        "max_tokens": 1000,
        "messages": messages
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if effort:
        kwargs["output_config"] = {"effort": effort}

    if stream:
        full_text = ""
        print("🤖 Assistant: ", end="", flush=True)
        t0 = time.perf_counter()
        n_events = n_text = 0
        first_event = first_text = None

        def dbg(msg):
            if DEBUG:
                print(f"\n[debug +{time.perf_counter() - t0:6.2f}s] {msg}", file=sys.stderr, flush=True)

        dbg(f"opening stream: model={kwargs['model']} effort={effort} max_tokens={kwargs['max_tokens']}")
        try:
            async with client.messages.stream(**kwargs) as s:
                async for event in s:
                    n_events += 1
                    if first_event is None:
                        first_event = time.perf_counter() - t0
                        dbg("first event received")
                    etype = event.type
                    if etype == "content_block_start":
                        dbg(f"block start: {event.content_block.type}")
                    elif etype == "content_block_stop":
                        dbg("block stop")
                    elif etype == "content_block_delta":
                        dtype = event.delta.type
                        if dtype == "text_delta":
                            n_text += 1
                            if first_text is None:
                                first_text = time.perf_counter() - t0
                                dbg("first text delta")
                            dbg(f"text_delta #{n_text} ({len(event.delta.text)} chars)") if DEBUG else None
                            print(event.delta.text, end="", flush=True)
                            full_text += event.delta.text
                        else:
                            # thinking_delta, signature_delta, input_json_delta, ...
                            if DEBUG and n_events % 20 == 0:
                                dbg(f"{dtype} (event #{n_events})")
                    elif etype == "message_delta":
                        dbg(f"message_delta stop_reason={event.delta.stop_reason}")
                    else:
                        dbg(f"event: {etype}")
        except Exception as e:
            print(f"\n[Stream error: {type(e).__name__}: {e}]", flush=True)
            if DEBUG:
                traceback.print_exc()
        dbg(f"done: {n_events} events, {n_text} text deltas, "
            f"first_event={first_event}, first_text={first_text}, total={time.perf_counter() - t0:.2f}s")
        return full_text
    else:
        message = await client.messages.create(**kwargs)
        text = "".join(b.text for b in message.content if b.type == "text")
        if not text:
            text = f"[No text returned (stop_reason={message.stop_reason}). Try a lower effort or higher max_tokens.]"
        return text