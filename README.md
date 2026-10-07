# You Should Know

A shareable Notch extension based on Nate Berkopec's Pi observer. It watches the text transcript without adding anything to the main agent's context:

1. TypeSafe **Jev 1.13.0** directly and cheaply decides whether an unacknowledged consequential problem is likely.
2. Only when `P(warn)` reaches the configured threshold does a fresh, tool-less Notch process turn the concern into a short **YSK** note.
3. The second stage uses Notch's configured `explore_model`; when that setting is empty, it uses the session's current provider/model.

The default threshold is `0.85`. Transcript snapshots are text-only and capped at 24,000 characters. The direct Jev request has no retries or provider fallback; the explanation child uses Notch's normal provider retry policy. Notes and observer usage metadata are persisted as extension-owned, non-context session entries.

**Data boundary:** every snapshot sent to TypeSafe includes recent user/assistant text, tool names and arguments, and textual tool results (including whether an empty result succeeded or failed). When Jev opens the gate, the same snapshot is also sent to the provider configured by `explore_model`. The extension does not redact credentials or other secrets that appear in that text or in tool arguments. Do not enable it for sessions whose transcript cannot be sent to both applicable providers.

## Requirements

- Python 3.10 or later.
- `TYPESAFE_API_KEY` must be present in the environment that launches Notch.
- A working Notch `explore_model` is recommended, for example in `~/.config/notch/config.json`:

  ```json
  {"explore_model": "openai/gpt-5.6-luna"}
  ```

The plugin process itself receives Notch's minimal environment. The Jev request is deliberately made by a worker launched through `host.exec`, so it inherits the main Notch environment without putting the TypeSafe key in argv. The explanation model also runs through a fresh Notch child and reuses Notch's user-level provider configuration and credentials; project settings, instructions, tools, extensions, and resources are excluded from that independent call.

## Install

Install directly from GitHub:

```sh
notch extensions install github:trobrock/notch-you-should-know
```

To update an existing installation:

```sh
notch extensions update notch-you-should-know
```

Restart Notch after installing or updating the package.

## Commands

```text
/ysk status
/ysk on
/ysk off
/ysk 0.90
/ysk dismiss
```

The footer shows cumulative Jev and explanation-model call counts and estimated costs. A `?` means the provider did not report enough information to calculate cost.

## Differences from the Pi version

This port uses the configured Notch explore model instead of hard-coding Pi's `openai/gpt-6-luna`. It does not include Pi's separate side-chat UI; dismissing and configuring the observer are available through `/ysk`.
