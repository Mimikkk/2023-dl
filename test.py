import numpy as np

if __name__ == '__main__':
  def lerp(a, b, t):
    return a + (b - a) * t

  values = [lerp(np.array([0, 0, 0]), np.array([0, 1, 0]), t) for t in np.linspace(0, 1, 10)]

  print(*[f"[{', '.join([f'{a:.2f}' for a in ar])}]" for ar in values], sep='\n')
