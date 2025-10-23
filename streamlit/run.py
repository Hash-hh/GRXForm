import subprocess
import sys
import os
import shutil


def main():
    print("Starting Molecule Design Visualizer...")
    print("Open your browser at http://localhost:8501")
    print("-" * 50)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_path = os.path.join(script_dir, "visualization.py")

    # Check if streamlit is available
    streamlit_path = shutil.which("streamlit")
    if streamlit_path is None:
        print("ERROR: streamlit command not found!")
        print("Please install streamlit: pip install streamlit")
        sys.exit(1)

    print(f"Using streamlit from: {streamlit_path}")

    # Run streamlit
    try:
        subprocess.run([
            "streamlit", "run", visualization_path,
            "--server.headless", "true"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down visualizer...")


if __name__ == "__main__":
    main()
