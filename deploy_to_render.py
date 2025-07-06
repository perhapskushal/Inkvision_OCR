#!/usr/bin/env python3
"""
Deployment helper script for Render
This script prepares your Flask app for Render deployment
"""

import os
import shutil
import sys

def backup_original_files():
    """Backup original files before replacing them"""
    print("Backing up original files...")
    
    if os.path.exists('app.py'):
        shutil.copy2('app.py', 'app_original.py')
        print("✓ Backed up app.py to app_original.py")
    
    if os.path.exists('requirements.txt'):
        shutil.copy2('requirements.txt', 'requirements_original.txt')
        print("✓ Backed up requirements.txt to requirements_original.txt")

def prepare_for_render():
    """Prepare files for Render deployment"""
    print("Preparing files for Render deployment...")
    
    # Backup original files
    backup_original_files()
    
    # Replace app.py with Render-compatible version
    if os.path.exists('app_render.py'):
        shutil.copy2('app_render.py', 'app.py')
        print("✓ Replaced app.py with Render-compatible version")
    else:
        print("❌ app_render.py not found!")
        return False
    
    # Replace requirements.txt with Render-compatible version
    if os.path.exists('requirements_render.txt'):
        shutil.copy2('requirements_render.txt', 'requirements.txt')
        print("✓ Replaced requirements.txt with Render-compatible version")
    else:
        print("❌ requirements_render.txt not found!")
        return False
    
    # Check if required files exist
    required_files = [
        'render.yaml',
        'build.sh',
        'templates/'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Make build.sh executable
    try:
        os.chmod('build.sh', 0o755)
        print("✓ Made build.sh executable")
    except Exception as e:
        print(f"⚠️  Warning: Could not make build.sh executable: {e}")
    
    return True

def restore_original_files():
    """Restore original files"""
    print("Restoring original files...")
    
    if os.path.exists('app_original.py'):
        shutil.copy2('app_original.py', 'app.py')
        os.remove('app_original.py')
        print("✓ Restored app.py")
    
    if os.path.exists('requirements_original.txt'):
        shutil.copy2('requirements_original.txt', 'requirements.txt')
        os.remove('requirements_original.txt')
        print("✓ Restored requirements.txt")

def check_dependencies():
    """Check if all required files exist"""
    required_files = [
        'app_render.py',
        'requirements_render.txt',
        'render.yaml',
        'build.sh',
        'templates/'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✓ All required files found")
    return True

def main():
    """Main function"""
    print("🚀 Render Deployment Helper")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python deploy_to_render.py prepare  # Prepare for Render")
        print("  python deploy_to_render.py restore  # Restore original files")
        print("  python deploy_to_render.py check    # Check dependencies")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'prepare':
        if check_dependencies():
            if prepare_for_render():
                print("\n✅ Files prepared for Render deployment!")
                print("\nNext steps:")
                print("1. Push your code to GitHub")
                print("2. Go to render.com and create a new Web Service")
                print("3. Connect your GitHub repository")
                print("4. Set Build Command to: ./build.sh")
                print("5. Set Start Command to: gunicorn app:app")
                print("6. Deploy!")
            else:
                print("\n❌ Failed to prepare files for deployment")
        else:
            print("\n❌ Missing required files. Cannot proceed.")
    
    elif command == 'restore':
        restore_original_files()
        print("\n✅ Original files restored!")
    
    elif command == 'check':
        check_dependencies()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'prepare', 'restore', or 'check'")

if __name__ == '__main__':
    main() 