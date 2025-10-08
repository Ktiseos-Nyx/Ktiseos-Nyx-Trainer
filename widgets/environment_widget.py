# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/environment_widget.py

import os
import shutil
import ipywidgets as widgets
from IPython.display import display


class EnvironmentWidget:
    def __init__(self, setup_manager, model_manager):
        self.setup_manager = setup_manager
        self.model_manager = model_manager
        self.backend_path = os.path.join(os.getcwd(), "trainer", "derrian_backend")
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the environment validation interface"""
        
        # Main header
        self.header = widgets.HTML("<h3>🔍 Environment Validation</h3>")
        
        # Status display
        self.status_display = widgets.HTML()
        
        # Diagnostic button
        self.diagnostic_button = widgets.Button(
            description="🔍 Run Full Diagnostics",
            button_style='info',
            tooltip="Run comprehensive system diagnostics"
        )
        self.diagnostic_button.on_click(self._run_diagnostics)
        
        # Diagnostic output
        self.diagnostic_output = widgets.Output()
        
        # Emergency setup (only shown if needed)
        self.emergency_setup = widgets.VBox()
        
        self.widget_box = widgets.VBox([
            self.header,
            self.status_display,
            self.diagnostic_button,
            self.diagnostic_output,
            self.emergency_setup
        ])
        
    def validate_environment(self):
        """Quick environment validation"""
        print("🎯 UNIFIED LORA TRAINER - Environment Check...")
        print("✨ Validating installation completed by installer.py")
        print()
        
        # Check if backend is properly installed
        if self._check_backend_installation():
            print("✅ Backend installation: OK")
            print("✅ Kohya sd_scripts: OK")
            print("✅ Ready for training!")
            print()
            print("💡 Environment is ready! Proceed to Cell 2 for training configuration.")
            
            self.status_display.value = """
            <div style='background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <strong>✅ Environment Status:</strong> Ready for training!
            </div>
            """
        else:
            print("⚠️ Backend not properly installed!")
            print("🔧 Run: python installer.py")
            print("📖 Or check installation documentation")
            
            self.status_display.value = """
            <div style='background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <strong>❌ Environment Status:</strong> Backend installation incomplete
            </div>
            """
            
            self._show_emergency_setup()
            
        print()
        print("💡 This notebook automatically detects your model type and uses:")
        print("   • SD 1.5/2.0 → train_network.py")
        print("   • SDXL → sdxl_train_network.py") 
        print("   • Flux → flux_train_network.py")
        print("   • SD3 → sd3_train_network.py")
        
    def _check_backend_installation(self):
        """Check if backend is properly installed"""
        return (os.path.exists(self.backend_path) and 
                os.path.exists(os.path.join(self.backend_path, "sd_scripts")))
                
    def _show_emergency_setup(self):
        """Show emergency setup widget if installation failed"""
        from widgets.setup_widget import SetupWidget
        
        emergency_desc = widgets.HTML("""
        <h4>🚨 Emergency Setup</h4>
        <p>Use this only if installer.py failed:</p>
        """)
        
        setup_widget = SetupWidget(self.setup_manager, self.model_manager)
        
        self.emergency_setup.children = [emergency_desc, setup_widget.widget_box]
        
    def _run_diagnostics(self, button):
        """Run comprehensive system diagnostics"""
        self.diagnostic_output.clear_output()
        
        with self.diagnostic_output:
            print("🔍 COMPREHENSIVE SYSTEM DIAGNOSTICS")
            print("=" * 50)
            
            # Use setup widget diagnostics
            from widgets.setup_widget import SetupWidget
            
            setup_widget = SetupWidget(self.setup_manager, self.model_manager)
            
            # Run container detection
            try:
                container_info = setup_widget._detect_container_environment()
                print(f"Environment: {container_info['environment']}")
                print(f"Provider: {container_info['provider_details']['name']}")
                print(f"GPU Count: {container_info['gpu_count']}")
                print(f"GPU Names: {container_info['gpu_names']}")
            except Exception as e:
                print(f"Container detection error: {e}")
            
            # Check key paths
            print("\n📁 PATH CHECKS:")
            paths_to_check = [
                ("Project Root", os.getcwd()),
                ("Backend", self.backend_path),
                ("SD Scripts", os.path.join(self.backend_path, "sd_scripts")),
                ("LyCORIS", os.path.join(self.backend_path, "lycoris")),
            ]
            
            for name, path in paths_to_check:
                status = "✅ EXISTS" if os.path.exists(path) else "❌ MISSING"
                print(f"   {name}: {status}")
                if os.path.exists(path):
                    print(f"      → {path}")
            
            # Check system dependencies
            print("\n🔧 SYSTEM DEPENDENCIES:")
            deps = ["python", "git", "aria2c"]
            for dep in deps:
                status = "✅ FOUND" if shutil.which(dep) else "❌ MISSING"
                print(f"   {dep}: {status}")
            
            # Check Python packages
            print("\n📦 KEY PYTHON PACKAGES:")
            packages = ["torch", "transformers", "accelerate", "diffusers", "bitsandbytes"]
            for pkg in packages:
                try:
                    __import__(pkg)
                    print(f"   {pkg}: ✅ INSTALLED")
                except ImportError:
                    print(f"   {pkg}: ❌ MISSING")
            
            print("\n✅ Diagnostic complete!")
            
    def display(self):
        """Display the widget"""
        self.validate_environment()
        display(self.widget_box)