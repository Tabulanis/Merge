"""
Every word of help in one place.

Written once here, rendered by the terminal (`forge help`), the web page
(/help), and the error messages that point at it. One source means the
answer you get is the same wherever you ask, and there's exactly one file
to fix when something changes.

Rules this content follows:
  * No jargon without a plain-language gloss right next to it.
  * Every "here's a problem" carries a "here's the exact fix" beside it.
  * Commands are copy-pasteable, whole, with no placeholders unless the
    placeholder is obvious (like sk-ant-YOUR-KEY).
  * Short. Someone reading help is stuck and slightly annoyed already.
"""

VERSION = "0.1"

# Shown once, the first time someone runs Forge with no config.
WELCOME = """Forge is a coding assistant that works in a folder on your computer.

You tell it what you want in plain English. It reads your files, changes
them, runs commands — asking permission before anything that changes your
stuff — and tells you what happened.

Two things worth knowing right now:

  1. It works in ONE folder at a time. Start it from the folder you care
     about, and that's all it can touch.

  2. It needs a brain. That's either a model running on your own machine
     (free, private) or Claude (costs money per use). Run `forge doctor`
     and it'll tell you what you've got and what's missing.

Type `forge help` any time. Nothing here can break your computer."""


TOPICS = {
    # ---------------------------------------------------------------- start
    "start": {
        "title": "Getting started",
        "blurb": "The shortest path to it actually doing something.",
        "body": """Open a terminal, go to a project folder, and start it:

    cd ~/my-project
    forge

You'll get a `›` prompt. Type what you want, like you'd say it to a person:

    make a python file that prints the first 20 prime numbers
    what's in this folder?
    the tests are failing, find out why

It'll show you what it's doing as it goes. When it wants to change a file
or run a command, it asks first — you type y or n.

To leave, press Ctrl-D or type /quit.

If nothing works at all, run `forge doctor`. It checks everything and tells
you exactly what to fix.""",
    },

    # --------------------------------------------------------------- models
    "models": {
        "title": "Models — the brain",
        "blurb": "Local and free, or Claude and paid. How to switch.",
        "body": """Forge doesn't think on its own. It sends your request to a model.

LOCAL MODELS (free, private, runs on your machine)
Start one first, then Forge:

    ~/forge/start-model.sh big      the main one, uses the graphics card
    ~/forge/start-model.sh tiny     small and fast, uses regular memory
    ~/forge/start-model.sh vision   the one that can look at pictures

The big one takes about 40 seconds to wake up. Leave it running; you only
start it once.

CLAUDE (costs money, per use)
You need a key from console.anthropic.com — this is separate from any
Claude subscription and is billed by usage. Then:

    export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY
    forge

SWITCHING
In the terminal:      /model            (lists them)
                      /model qwen30b    (switches)
In the browser:       click one in the list

A model switched in the browser reaches the terminal on your next message.
No restart.""",
    },

    # ----------------------------------------------------------- permission
    "permission": {
        "title": "Permission — staying in control",
        "blurb": "When it asks before doing things, and how to change that.",
        "body": """Reading files is always allowed. Changing a file or running a
command asks you first, and shows you exactly what it wants to do.

Three modes:

    ask     asks before each change          (default, safest)
    auto    just does it                     (fast, for scratch work)
    deny    reads only, changes nothing      (safe browsing)

Change it with `/perm auto` in the terminal, or the buttons in the browser.

A note on `auto`: it's genuinely useful when you're watching, and genuinely
risky when you're not. If you walk away mid-task in auto mode, it will keep
going without you.""",
    },

    # ---------------------------------------------------------------- phone
    "phone": {
        "title": "Using it from a phone or tablet",
        "blurb": "Same assistant, on the couch.",
        "body": """On the computer, start the web version:

    forge-dash --network

It prints two links and a token. Open the one with your computer's address
on your phone — it looks like:

    http://192.168.0.11:8770/chat?token=SOMELONGSTRING

Both devices have to be on the same wifi.

The token is a password. It's there because this thing can run commands on
your computer, and without it anyone on your network could do the same. The
page remembers it, so you only paste that link once.

Three pages:
    /chat      talk to it
    /bricks    build multi-model workflows by stacking blocks
    /          pick models, change settings

Lost the link? Run `forge-dash --network` again and it prints it. Want a
fresh token (if you shared the old one): `forge-dash --new-token`.""",
    },

    # ------------------------------------------------------------- commands
    "commands": {
        "title": "Slash commands",
        "blurb": "The / things you can type in the terminal.",
        "body": """    /help              this
    /help <topic>      a specific topic, e.g. /help models
    /doctor            check everything and report problems
    /model             list models, or /model <name> to switch
    /perm <mode>       ask · auto · deny
    /media             what it can see and hear right now
    /flows             list the multi-model recipes
    /run <flow> <task> run a recipe
    /clear             forget the conversation, keep the settings
    /config            where your settings file lives
    /tools             what it's able to do
    /sessions          your saved chats — the same list the web page shows
    /resume <number>   pick an old chat up where it left off
    /off               leave AND stop the local models (frees the GPU)
    /quit              leave

Anything not starting with / is a message to the assistant.

Every chat is saved automatically. `merge -c` in a folder continues your
last chat there; the web page's drawer shows the same chats, so you can
start at the desk and finish from the couch.""",
    },

    # ---------------------------------------------------------------- flows
    "flows": {
        "title": "Recipes — chaining models together",
        "blurb": "Why a loop beats a bigger model.",
        "body": """A recipe is a few steps run in order. The useful ones have a loop
that keeps going until a real check passes.

    /flows                                  see what exists
    /run build-until-it-works add a login    run one

The ones that ship:

    build-until-it-works   builds it, runs the tests, and if they fail feeds
                           the real error back and tries again
    write-and-review       one model writes, another reviews, the first fixes
    route-by-difficulty    a small model decides, the big one does the work
    explore-then-change    look around the code first, then edit
    fix-what-you-see       look at a screenshot, fix what's visually wrong
    do-what-i-said         transcribe a voice memo and carry it out

Why loops matter: a smaller local model is far better at "run the tests, read
the failure, fix it" than at getting it right first time. The exit condition
is a command's exit code, not the model's opinion of its own work — which is
exactly the thing smaller models get wrong.

Build your own by stacking blocks at /bricks in the browser.""",
    },

    # ---------------------------------------------------------------- media
    "media": {
        "title": "Seeing and hearing",
        "blurb": "Screenshots, images, voice.",
        "body": """Optional extras. Forge works fine without any of them, and only
offers a tool when the thing behind it actually works.

    /media     shows what's live right now

SEEING     Start the vision model: `~/forge/start-model.sh vision`
           Then it can look at screenshots and images and tell you what's
           wrong with them. Slow — a couple of minutes per picture — so it's
           for "look at this bug", not conversation.

HEARING    Point it at a voice recording and it types it out. Fast, about
           two seconds.

SPEAKING   It can say short things out loud, handy when you've walked away.

The vision model wants the graphics card that the big coding model is
already using, so in practice you stop one to run the other.""",
    },

    # -------------------------------------------------------------- trouble
    "trouble": {
        "title": "When something's wrong",
        "blurb": "The five things that actually go wrong, and their fixes.",
        "body": """Run `forge doctor` first. It checks all of this and tells you the
fix. The common ones:

"Model call failed" / it just hangs
    The model isn't running. Start it:  ~/forge/start-model.sh big
    Give it 40 seconds — a big model takes a while to wake up.

"No Anthropic API key found"
    You're pointed at Claude but haven't given it a key. Either switch to a
    local model (`/model qwen30b`) or set the key (see `forge help models`).

It says it did something, but nothing changed
    Local models sometimes claim work they didn't do. Ask it to show you:
    "read the file back to me". If it made that up, say so — it corrects.
    The build-until-it-works recipe defends against this properly, because
    a command's exit code can't be talked into agreeing.

The phone can't reach it
    Both devices on the same wifi? Started with `--network`, not plain
    `forge-dash`? Using the link with the token on the end?

It's being slow
    Local models are slower than cloud ones, and the first message after
    starting is slowest while the model warms up. A 30B model on a good
    graphics card runs a few seconds per reply. If it's much worse than
    that, the model may have loaded into regular memory instead of the
    graphics card — check `nvidia-smi`.""",
    },

    # --------------------------------------------------------------- safety
    # ---------------------------------------------------------------- power
    "power": {
        "title": "Turning her off and on",
        "blurb": "Get your graphics card back when you're done.",
        "body": """The local models hold the graphics card the whole time they run —
the big one alone keeps ~23GB of it. When you're done for now:

    forge off        stop every local model, free the card
    forge on         start the big one again (takes about a minute to load)
    forge on vision  start a specific one: big, vision, little, tiny

Inside a chat, /off does the same and says goodbye.

The dashboard stays up — it uses no graphics memory, and its Power card
has the same off/on buttons, so you can wake her from the browser or a
phone without touching a terminal. The models also come back on their own
next time the computer boots.""",
    },

    "safety": {
        "title": "Is this safe?",
        "blurb": "What it can and can't do to your computer.",
        "body": """What it CAN do:
    Read, create, change, and delete files in the folder you started it in.
    Run commands as you, in that folder.

What it CAN'T do:
    Touch anything outside that folder. Paths that try to escape are
    refused, and the refusal is shown to the model so it corrects itself.

What protects you:
    It asks before changing anything (unless you set `auto`).
    Everything happens in one folder you chose.
    With a local model, nothing you type or any of your code leaves your
    machine at all.

Worth doing anyway: work in a folder that's in git. Then anything it does
is one `git diff` away from being seen and one `git checkout` away from
being undone. That's a better safety net than any permission prompt.""",
    },
}

# Order they appear in listings — start first, trouble near the end.
ORDER = ["start", "models", "permission", "commands", "phone",
         "flows", "media", "power", "trouble", "safety"]


def topic(name: str):
    """Find a topic by name or unambiguous prefix. Returns (key, data) or None."""
    name = (name or "").strip().lower()
    if name in TOPICS:
        return name, TOPICS[name]
    hits = [k for k in ORDER if k.startswith(name)]
    if len(hits) == 1:
        return hits[0], TOPICS[hits[0]]
    return None


def search(query: str):
    """Topics whose text mentions every word of the query."""
    words = [w for w in (query or "").lower().split() if w]
    if not words:
        return []
    out = []
    for key in ORDER:
        t = TOPICS[key]
        hay = (key + " " + t["title"] + " " + t["blurb"] + " " + t["body"]).lower()
        if all(w in hay for w in words):
            out.append((key, t))
    return out
