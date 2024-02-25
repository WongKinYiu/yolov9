import click
from questionary import text, confirm
import yaml

# Existing code ...

def run_interactive():
    weights = text("Enter model path(s):").ask(default=str(ROOT / 'yolov5s-cls.pt'))
    source = text("Enter file/dir/URL/glob/screen/0(webcam):").ask(default=str(ROOT / 'data/images'))
    # ... other interactive prompts ...

    # ... your existing logic ...

def run_with_config(config_path='config.yaml'):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Use config values instead of command-line args
    run(**config)

@click.command()
@click.option('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model path(s)')
@click.option('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
# ... other options ...
@click.option('--interactive', is_flag=True, help='Run in interactive mode')
@click.option('--config-path', type=str, default='config.yaml', help='Path to configuration file')
def run_cli(weights, source, data, imgsz, device, view_img, save_txt, nosave, augment, visualize, update, project, name, exist_ok, half, dnn, vid_stride, interactive, config_path):
    if interactive:
        run_interactive()
    elif config_path:
        run_with_config(config_path)
    else:
        # ... your existing logic ...

if __name__ == '__main__':
    run_cli()
