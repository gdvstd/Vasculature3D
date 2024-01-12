#!/bin/bash

# Find Python processes
python_processes=$(pgrep -x "python")

# Check if any Python processes are running
if [ -n "$python_processes" ]; then
    echo "Python processes found:"
    ps -p $python_processes #-o pid,cmd

    echo -e "\nSpecify the PID(s) of the Python process(es) to kill (comma-separated), or enter 'exit' to end:"
    
    read user_input
    
    if [ "$user_input" == "exit" ]; then
        exit
    else
        # Kill specific Python processes based on user input
        IFS=',' read -ra pids <<< "$user_input"
        for pid in "${pids[@]}"; do
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                # Check if the PID is valid
                if ps -p $pid > /dev/null; then
                    kill -9 $pid
                    echo "Python process with PID $pid killed."
                else
                    echo "Invalid PID: $pid. No action taken."
                fi
            else
                echo "Invalid input: $pid. Please enter a valid PID or 'all'."
            fi
        done
    fi
else
    echo "No Python processes found."
fi

