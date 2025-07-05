#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)
PYTHON_PATH="$CURRENT_DIR/.venv/bin/python"
SCRIPT_PATH="$CURRENT_DIR/options_strategy_monitor.py"

echo "Setting up daily cron job for Bitcoin Treasury Options Monitor..."
echo "Current directory: $CURRENT_DIR"
echo "Python path: $PYTHON_PATH"
echo "Script path: $SCRIPT_PATH"

# Check if the Python executable exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Error: Python executable not found at $PYTHON_PATH"
    echo "Please make sure your virtual environment is set up correctly."
    exit 1
fi

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Create the cron job entry
CRON_JOB="0 8 * * * $PYTHON_PATH $SCRIPT_PATH"

echo ""
echo "Cron job to be added:"
echo "$CRON_JOB"
echo ""

# Ask for confirmation
read -p "Do you want to add this cron job? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add the cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    
    echo "✅ Cron job added successfully!"
    echo ""
    echo "The script will now run daily at 8:00 AM."
    echo "You can view your current cron jobs with: crontab -l"
    echo "You can edit your cron jobs with: crontab -e"
    echo "You can remove all cron jobs with: crontab -r"
    echo ""
    echo "To test the script now, run:"
    echo "$PYTHON_PATH $SCRIPT_PATH"
else
    echo "❌ Cron job not added."
fi 