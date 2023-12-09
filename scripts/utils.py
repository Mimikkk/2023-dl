import sys

def in_venv():
  return (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

codes = {
  'black': 30,
  'red': 31,
  'green': 32,
  'yellow': 33,
  'blue': 34,
  'magenta': 35,
  'cyan': 36,
  'white': 37
}

def chalk(text: str, color: str | None = None) -> None:
  if color is None: return text
  return f"\033[{codes[color]}m{text}\033[00m"

def cprint(prefix: str, text: str, color: str = None, *, prefix_color: str) -> None:
  print(f'[{chalk(prefix, prefix_color)}] {chalk(text, color)}')
