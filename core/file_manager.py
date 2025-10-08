# core/file_manager.py
import os
import socket
import sys
import threading
import time
from typing import Optional


class FileManagerUtility:
    """File manager utility using IMjoy Elfinder for Jupyter notebooks"""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root or os.getcwd()
        self.active_managers = {}  # Track running file managers

    def _find_free_port(self, start_port: int = 8765) -> int:
        """Find a free port starting from start_port"""
        port = start_port
        while port < start_port + 100:  # Try 100 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"No free port found between {start_port} and {start_port + 100}")

    def _start_elfinder_server(self, root_dir: str, port: int):
        """Start the IMjoy Elfinder server in a separate thread"""
        try:
            # Import here to avoid dependency issues if not installed
            from imjoy_elfinder.app import main
            main([f"--root-dir={root_dir}", f"--port={port}"])
        except ImportError:
            print("❌ IMjoy Elfinder not installed. Install with: pip install imjoy-elfinder")
            return False
        except Exception as e:
            print(f"❌ Error starting file explorer: {str(e)}")
            return False

    def start_file_manager(self, root_dir: Optional[str] = None, open_in_new_tab: bool = False,
                          port: Optional[int] = None) -> bool:
        """
        Start file manager with IMjoy Elfinder
        
        Args:
            root_dir: Directory to browse (defaults to project root)
            open_in_new_tab: Whether to open in new tab vs embedded iframe
            port: Port to use (auto-finds free port if None)
        """
        if root_dir is None:
            root_dir = self.project_root

        if not os.path.exists(root_dir):
            print(f"❌ Directory not found: {root_dir}")
            return False

        # Find free port if not specified
        if port is None:
            try:
                port = self._find_free_port()
            except RuntimeError as e:
                print(f"❌ {e}")
                return False

        # Check if already running on this port
        if port in self.active_managers:
            print(f"📁 File manager already running on port {port}")
            return True

        print(f"🚀 Starting file manager for: {root_dir}")
        print(f"🌐 Port: {port}")
        print(f"📱 Mode: {'New tab' if open_in_new_tab else 'Embedded iframe'}")

        # Start server in background thread
        thread = threading.Thread(
            target=self._start_elfinder_server,
            args=[root_dir, port],
            daemon=True
        )
        thread.start()

        # Give server time to start
        time.sleep(2)

        # Store active manager info
        self.active_managers[port] = {
            'root_dir': root_dir,
            'thread': thread,
            'new_tab': open_in_new_tab
        }

        # Display file manager
        return self._display_file_manager(port, open_in_new_tab)

    def _display_file_manager(self, port: int, open_in_new_tab: bool) -> bool:
        """Display the file manager in Jupyter"""
        try:
            # Check if we're in Colab or Jupyter
            if 'google.colab' in sys.modules:
                # Google Colab
                from google.colab import output
                if open_in_new_tab:
                    output.serve_kernel_port_as_window(port)
                else:
                    output.serve_kernel_port_as_iframe(port, height="600")

            elif 'IPython' in sys.modules:
                # Regular Jupyter
                from IPython.display import HTML, IFrame, display

                if open_in_new_tab:
                    # For Jupyter, we'll show a link and try to open in new tab
                    url = f"http://localhost:{port}"
                    html = f"""
                    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                        <h4>📁 File Manager</h4>
                        <p>Click to open file manager in new tab:</p>
                        <a href="{url}" target="_blank" style="
                            display: inline-block;
                            padding: 10px 20px;
                            background-color: #007cba;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                            font-weight: bold;
                        ">🚀 Open File Manager</a>
                        <p style="margin-top: 10px; font-size: 12px; color: #666;">
                            URL: <code>{url}</code>
                        </p>
                    </div>
                    """
                    display(HTML(html))
                else:
                    # Embedded iframe
                    url = f"http://localhost:{port}"
                    iframe = IFrame(src=url, width="100%", height="600")
                    display(iframe)

            else:
                # Fallback - just print URL
                print(f"🌐 File manager running at: http://localhost:{port}")

            print("✅ File manager started successfully!")
            print(f"💡 Use stop_file_manager({port}) to stop this instance")
            return True

        except Exception as e:
            print(f"❌ Error displaying file manager: {str(e)}")
            return False

    def stop_file_manager(self, port: int) -> bool:
        """Stop file manager on specific port"""
        if port not in self.active_managers:
            print(f"❌ No file manager running on port {port}")
            return False

        try:
            # Note: IMjoy Elfinder doesn't have a clean shutdown method
            # The thread will be daemon so it'll close when Python exits
            del self.active_managers[port]
            print(f"✅ Stopped file manager on port {port}")
            return True
        except Exception as e:
            print(f"❌ Error stopping file manager: {str(e)}")
            return False

    def stop_all_managers(self):
        """Stop all running file managers"""
        ports = list(self.active_managers.keys())
        for port in ports:
            self.stop_file_manager(port)
        print("✅ Stopped all file managers")

    def list_active_managers(self):
        """List all active file managers"""
        if not self.active_managers:
            print("📁 No active file managers")
            return

        print("📁 Active file managers:")
        for port, info in self.active_managers.items():
            mode = "New tab" if info['new_tab'] else "Embedded"
            print(f"  🌐 Port {port}: {info['root_dir']} ({mode})")

    def start_project_browser(self, open_in_new_tab: bool = True) -> bool:
        """Start file manager for the main project directory"""
        return self.start_file_manager(
            root_dir=self.project_root,
            open_in_new_tab=open_in_new_tab
        )

    def start_dataset_browser(self, dataset_dir: str, open_in_new_tab: bool = True) -> bool:
        """Start file manager for a specific dataset directory"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        return self.start_file_manager(
            root_dir=dataset_path,
            open_in_new_tab=open_in_new_tab
        )

    def start_output_browser(self, open_in_new_tab: bool = True) -> bool:
        """Start file manager for the output directory"""
        output_path = os.path.join(self.project_root, "output")
        return self.start_file_manager(
            root_dir=output_path,
            open_in_new_tab=open_in_new_tab
        )

# Global file manager instance
_file_manager = None

def get_file_manager():
    """Get or create global file manager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManagerUtility()
    return _file_manager
