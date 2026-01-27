import fire
import subprocess

class Pipeline:
    """A class to manage the DVC pipeline stages through a CLI."""

    def pull(self):
        """Pulls data from DVC remote storage."""
        try:
            print("--- DVC STAGE: PULL DATA ---")
            subprocess.run(["dvc", "pull"], check=True)
            print("DVC pull complete.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during DVC pull: {e}")
            raise
        except FileNotFoundError:
            print("The 'dvc' command was not found. Ensure DVC is installed and in your PATH.")
            raise

    def train(self):
        """Runs the training stage."""
        print("Starting training...")
        # Since run_training is decorated with hydra, it's best to call it
        # via a subprocess to ensure hydra initializes correctly.
        subprocess.run(["python", "defects_in_welds/training/run_training.py"], check=True)
        print("Training complete.")

if __name__ == '__main__':
    fire.Fire(Pipeline)
