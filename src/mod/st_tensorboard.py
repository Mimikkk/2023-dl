import streamlit.components.v1 as components
print("importing tensorboard...")
from tensorboard import manager
import shlex
from pathlib import Path


def st_tensorboard(
    logdir: str = "./resources/logs/",
    port: int = 6006,
    width: int | None = None,
    height: int = 780,
    scrolling: bool = True,
    frame_id: str = "tensorboard-frame"
):
  """
  Start a TensorBoard instance and display it in Streamlit.

  :param logdir: The directory containing the TensorBoard logs. Defaults to "./resources/logs/".
  :param port: The port number for TensorBoard. Defaults to 6006.
  :param width: The width of the TensorBoard iframe. Defaults to None.
  :param height: The height of the TensorBoard iframe. Defaults to 780.
  :param scrolling: Whether to enable scrolling in the TensorBoard iframe. Defaults to True.
  :param frame_id: The ID of the TensorBoard iframe. Defaults to "tensorboard-frame".
  :return: The HTML code for embedding the TensorBoard iframe in a Streamlit app.
  """

  logdir = Path(str(logdir)).as_posix()

  instance = manager.start(shlex.split(f"--logdir {logdir} --port {port}", comments=True, posix=True))
  if isinstance(instance, manager.StartReused):
    port = instance.info.port
    print(f"Reusing TensorBoard on port {port}")
  else:
    print(f"Starting TensorBoard on port {port}")

  return components.html(
    f"""
    <iframe id="{frame_id}" width="100%" height="{height}" frameborder="0" src="http://localhost:{port}"></iframe>
    """,
    width=width,
    height=height,
    scrolling=scrolling
  )
