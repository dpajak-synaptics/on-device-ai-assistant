#!/bin/bash

# Define ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print messages in color
function print_message() {
    echo -e "${1}${2}${NC}"
}

# Save the current directory
CURRENT_DIR=$(pwd)

# Change directory to 'models'
mkdir -p models
print_message $YELLOW "Changing directory to 'models'..."
cd models || { print_message $RED "Failed to change directory to 'models'. Exiting."; exit 1; }

# Install Piper speech-to-text
print_message $GREEN "Installing Rhasspy Piper..."
print_message $GREEN "==============================\n"
# Download Piper tarball
curl -L -O https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz || { print_message $RED "Failed to download Piper tarball. Exiting."; exit 1; }
# Extract the tarball
tar -xzf piper_linux_aarch64.tar.gz || { print_message $RED "Failed to extract Piper tarball. Exiting."; exit 1; }
# Remove the tarball after extraction
rm piper_linux_aarch64.tar.gz
# Change directory to 'piper'
cd piper || { print_message $RED "Failed to change directory to 'piper'. Exiting."; exit 1; }


# Return to the previous directory
cd "$CURRENT_DIR" || { print_message $RED "Failed to return to previous directory. Exiting."; exit 1; }

cd models

print_message $GREEN "Installing Llama.cpp embedding..."
print_message $GREEN "=================================\n"
git clone --branch master --single-branch https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build # -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a
cmake --build build --config release --target llama-embedding
#cmake --build build --config release --target llama-cli

cd "$CURRENT_DIR"

# Check if .venv directory exists
if [ ! -d ".venv" ]; then
    print_message $GREEN "Virtual environment not found. Creating one..."
    python3 -m venv .venv || { print_message $RED "Failed to create virtual environment. Exiting."; exit 1; }
fi

# Activate the virtual environment
source .venv/bin/activate || { print_message $RED "Failed to activate virtual environment. Exiting."; exit 1; }

# Install required Python packages, using the path from $CURRENT_DIR
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt" || { print_message $RED "Failed to install Python packages. Exiting."; exit 1; }
else
    print_message $RED "requirements.txt not found in $CURRENT_DIR. Exiting."
    exit 1
fi

mkdir -p cached

# Prompt the user
print_message $GREEN "\nInstallation complete."
read -p "Do you want to install the On Device Assistant service to run on boot? (y/n) [default: n]: " user_input
user_input=${user_input:-n}

# Check the response
if [[ "$user_input" =~ ^[Yy]$ ]]; then
    echo "Installing the On Device Assistant service..."

    # Define paths
    SCRIPT_PATH="/home/root/on-device-assistant/main.py"
    SERVICE_PATH="/etc/systemd/system/on-device-assistant.service"

    # Make sure the Python script is executable
    chmod +x "$SCRIPT_PATH"

    # Create the systemd service file
    echo "[Unit]
Description=On Device Assistant
After=network.target

[Service]
ExecStart=/bin/bash -c 'source /home/root/on-device-assistant/.venv/bin/activate && /home/root/on-device-assistant/.venv/bin/python3 /home/root/on-device-assistant/main.py'
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
" > "$SERVICE_PATH"

    # Set the correct permissions for the service file
    chmod 644 "$SERVICE_PATH"

    # Reload systemd to pick up the new service
    systemctl daemon-reload

    # Enable the service to start on boot
    systemctl enable on-device-assistant.service

    # Start the service immediately
    systemctl start on-device-assistant.service

    echo "Service has been installed. It will run on boot."
else
    echo "Service installation skipped."
fi


# Print completion message
print_message $GREEN "Run the following command to start the assistant:\n"
print_message $GREEN "source .venv/bin/activate\npython3 main.py"
