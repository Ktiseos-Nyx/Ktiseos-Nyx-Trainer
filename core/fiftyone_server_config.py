# core/fiftyone_server_config.py
"""
FiftyOne Server Configuration for Remote Environments
Handles common server deployment issues with FiftyOne in containers.
"""

import os
import socket
import time
from typing import Any, Dict, Optional


def detect_server_environment() -> Dict[str, Any]:
    """Detect if we're running in a server environment and get network info"""
    info = {
        'is_remote_server': False,
        'is_jupyter_lab': False,
        'is_container': False,
        'jupyter_base_url': '',
        'external_ip': None,
        'internal_ip': None,
        'available_ports': [],
        'fiftyone_config': {}
    }

    # Check if we're in Jupyter Lab
    if os.environ.get('JUPYTER_ENABLE_LAB') or 'lab' in os.environ.get('JUPYTER_CONFIG_DIR', ''):
        info['is_jupyter_lab'] = True

    # Check if we're in a container
    if os.path.exists('/.dockerenv') or os.environ.get('CONTAINER'):
        info['is_container'] = True
        info['is_remote_server'] = True

    # Check for remote server indicators
    remote_indicators = [
        os.environ.get('SSH_CLIENT'),
        os.environ.get('SSH_CONNECTION'),
        '/workspace' in os.getcwd(),
        os.environ.get('RUNPOD_POD_ID'),
        os.environ.get('VAST_CONTAINERLABEL')
    ]
    if any(remote_indicators):
        info['is_remote_server'] = True

    # Get IP addresses
    try:
        # Internal IP (container/private network)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            info['internal_ip'] = s.getsockname()[0]
    except Exception:
        pass

    # Try to get external IP from environment variables (common in cloud containers)
    external_ip_sources = [
        os.environ.get('RUNPOD_PUBLIC_IP'),
        os.environ.get('VAST_PUBLIC_IP'),
        os.environ.get('PUBLIC_IP')
    ]
    for ip in external_ip_sources:
        if ip:
            info['external_ip'] = ip
            break

    # Get Jupyter base URL if available
    info['jupyter_base_url'] = os.environ.get('JUPYTER_BASE_URL', '')

    return info


def find_available_port(start_port: int = 5151, max_attempts: int = 50) -> Optional[int]:
    """Find an available port for FiftyOne app"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def configure_fiftyone_for_server(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """Configure FiftyOne settings for server environments"""
    import fiftyone as fo

    config = {}

    if server_info['is_remote_server']:
        # Find available port
        port = find_available_port()
        if not port:
            raise RuntimeError("Could not find available port for FiftyOne")

        config['port'] = port

        # Configure for remote access
        if server_info['is_container']:
            # Container environment - bind to all interfaces
            config['address'] = '0.0.0.0'
        else:
            # Regular server - use internal IP
            config['address'] = server_info.get('internal_ip', '127.0.0.1')

        # Set FiftyOne config
        fo.config.desktop_app = False  # Force web app mode
        fo.config.do_not_track = True  # Disable analytics in server environments

        print("🌐 FiftyOne configured for server environment:")
        print(f"   - Address: {config['address']}:{config['port']}")

        if server_info['external_ip']:
            external_url = f"http://{server_info['external_ip']}:{config['port']}"
            print(f"   - External URL: {external_url}")
            config['external_url'] = external_url

        # Configure for Jupyter Lab if applicable
        if server_info['is_jupyter_lab']:
            print("   - Jupyter Lab detected: Use browser tab instead of sidecar")
            config['jupyter_lab_mode'] = True
    else:
        # Local environment - use defaults
        config['address'] = '127.0.0.1'
        config['port'] = 5151

    return config


def launch_fiftyone_session(dataset, server_config: Dict[str, Any], auto_open: bool = False):
    """Launch FiftyOne session with server-optimized configuration"""
    import fiftyone as fo

    # Apply server configuration
    launch_params = {
        'port': server_config.get('port', 5151),
        'address': server_config.get('address', '127.0.0.1'),
        'remote': server_config.get('external_url') is not None,
        'auto': auto_open  # Usually False for servers
    }

    # Launch session
    session = fo.launch_app(dataset, **launch_params)

    # Print access information
    if server_config.get('external_url'):
        print(f"🌍 FiftyOne App URL: {server_config['external_url']}")
        print("   Copy this URL to access FiftyOne from your browser")
    elif server_config.get('jupyter_lab_mode'):
        local_url = f"http://{launch_params['address']}:{launch_params['port']}"
        print(f"🔗 FiftyOne App URL: {local_url}")
        print("   Open this URL in a new browser tab")

    return session


def wait_for_fiftyone_ready(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for FiftyOne app to be ready to accept connections"""
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
            continue

    return False


def create_server_friendly_dataset_launcher():
    """Create a dataset launcher optimized for server environments"""

    def launch_with_server_config(dataset_path: str):
        """Launch FiftyOne with automatic server detection and configuration"""
        try:
            import fiftyone as fo
        except ImportError:
            print("❌ FiftyOne not available. Install with: pip install fiftyone")
            return None

        # Detect server environment
        server_info = detect_server_environment()

        # Configure FiftyOne for the detected environment
        try:
            server_config = configure_fiftyone_for_server(server_info)
        except Exception as e:
            print(f"⚠️ Server configuration failed: {e}")
            print("Falling back to default configuration...")
            server_config = {'address': '127.0.0.1', 'port': 5151}

        # Create dataset
        print(f"📁 Loading dataset from: {dataset_path}")
        dataset = fo.Dataset.from_images_dir(dataset_path, recursive=True)

        # Add Kohya-specific metadata
        print("🏷️ Adding Kohya folder metadata...")
        for sample in dataset:
            folder_name = os.path.basename(os.path.dirname(sample.filepath))
            if '_' in folder_name:
                parts = folder_name.split('_', 1)
                sample['repeats'] = int(parts[0])
                sample['concept'] = parts[1]
                sample.save()

        # Launch session
        print("🚀 Launching FiftyOne session...")
        session = launch_fiftyone_session(dataset, server_config, auto_open=False)

        # Wait for app to be ready
        if wait_for_fiftyone_ready(
            server_config['address'],
            server_config['port']
        ):
            print("✅ FiftyOne is ready!")
        else:
            print("⚠️ FiftyOne may still be starting up...")

        return session, server_config

    return launch_with_server_config
