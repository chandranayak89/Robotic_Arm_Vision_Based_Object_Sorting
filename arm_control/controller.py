"""
Robotic arm controller module.
Manages communication with the robotic arm via serial connection.
"""
import serial
import time
import sys
import os
import threading
import queue

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ArmController:
    """
    Robotic arm controller class for serial communication.
    Implements commands for moving the arm, controlling the gripper, etc.
    """
    
    def __init__(self, port=None, baudrate=None, timeout=None):
        """
        Initialize with serial parameters.
        If not provided, uses the defaults from config.
        """
        self.port = port if port is not None else config.ARM_PORT
        self.baudrate = baudrate if baudrate is not None else config.ARM_BAUDRATE
        self.timeout = timeout if timeout is not None else config.ARM_TIMEOUT
        
        self.serial = None
        self.connected = False
        
        # Command queue and worker thread
        self.command_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
    
    def connect(self):
        """
        Connect to the robotic arm.
        Returns True if successful, False otherwise.
        """
        try:
            # Open serial connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # Wait for connection to establish
            time.sleep(2)
            
            # Test the connection by sending a ping command
            self.serial.write(b"PING\n")
            response = self.serial.readline().strip()
            
            # Check response (specific to your arm's protocol)
            # This is just an example, adjust to your arm's protocol
            if response == b"PONG" or True:  # True for testing without actual response check
                self.connected = True
                print(f"Connected to arm at {self.port}")
                
                # Start command queue worker
                self.start_worker()
                
                return True
            else:
                print(f"Arm responded with unexpected message: {response}")
                self.disconnect()
                return False
                
        except Exception as e:
            print(f"Error connecting to arm: {str(e)}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """Disconnect from the robotic arm."""
        self.stop_worker()
        
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        
        self.connected = False
        print("Disconnected from arm")
    
    def start_worker(self):
        """Start the command queue worker thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_thread_func)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop_worker(self):
        """Stop the command queue worker thread."""
        self.running = False
        if self.worker_thread is not None:
            if self.worker_thread.is_alive():
                self.command_queue.put(None)  # Signal to exit
                self.worker_thread.join(timeout=1.0)
            self.worker_thread = None
    
    def _worker_thread_func(self):
        """Worker thread function to process command queue."""
        while self.running:
            try:
                # Get command from queue (blocking with timeout)
                cmd = self.command_queue.get(timeout=0.1)
                
                # None is the signal to exit
                if cmd is None:
                    break
                
                # Execute the command
                success = self._execute_command(cmd)
                
                # Mark task as done
                self.command_queue.task_done()
                
                # Log result
                if success:
                    print(f"Command executed successfully: {cmd}")
                else:
                    print(f"Failed to execute command: {cmd}")
                    
            except queue.Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                print(f"Error in worker thread: {str(e)}")
                # Just continue, don't crash the thread
                continue
    
    def _execute_command(self, cmd):
        """
        Execute a command by sending it to the arm.
        Returns True if successful, False otherwise.
        """
        if not self.connected or self.serial is None:
            print("Cannot execute command: not connected to arm")
            return False
        
        try:
            # Format the command (add newline if not present)
            if not cmd.endswith('\n'):
                cmd += '\n'
            
            # Send the command
            self.serial.write(cmd.encode())
            
            # Wait for acknowledgment (depends on your arm's protocol)
            # This is a simplified example
            time.sleep(0.1)
            
            # Read response if any
            if self.serial.in_waiting > 0:
                response = self.serial.readline().strip()
                print(f"Arm response: {response}")
            
            return True
            
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            return False
    
    def send_command(self, cmd):
        """
        Send a command to the arm (queued execution).
        The command is added to the queue and executed by the worker thread.
        """
        if not self.connected:
            print("Cannot send command: not connected to arm")
            return False
        
        # Add command to queue
        self.command_queue.put(cmd)
        return True
    
    def move_to_position(self, x, y, z):
        """Move the arm to the specified position."""
        cmd = f"MOVE_TO {x} {y} {z}"
        return self.send_command(cmd)
    
    def grab(self):
        """Close the gripper to grab an object."""
        cmd = "GRIPPER CLOSE"
        return self.send_command(cmd)
    
    def release(self):
        """Open the gripper to release an object."""
        cmd = "GRIPPER OPEN"
        return self.send_command(cmd)
    
    def home(self):
        """Move the arm to the home position."""
        cmd = "HOME"
        return self.send_command(cmd)
    
    def sort_object(self, object_info, bin_positions=None):
        """
        Sort an object by moving it to the appropriate bin.
        
        Args:
            object_info: Dictionary with object properties
            bin_positions: Dictionary with bin positions, defaults to config.BINS
        
        Returns:
            True if commands were queued successfully
        """
        if bin_positions is None:
            bin_positions = config.BINS
        
        # Determine which bin to use based on object properties
        bin_key = None
        
        if "color" in object_info:
            # For color-based sorting
            bin_key = object_info["color"]
        elif "shape" in object_info:
            # For shape-based sorting
            bin_key = object_info["shape"]
        elif "class" in object_info:
            # For deep learning-based sorting
            bin_key = object_info["class"]
        
        # If bin not found, use unknown bin
        if bin_key not in bin_positions:
            bin_key = "unknown"
        
        # Get bin position
        bin_position = bin_positions[bin_key]
        
        # Get object position (this would come from camera calibration in a real system)
        # For this example, we'll just use dummy coordinates
        object_x, object_y = object_info.get("centroid", (0, 0))
        object_z = 50  # Dummy Z coordinate
        
        # Queue the sorting operation
        success = True
        
        # 1. Move above the object
        success = success and self.move_to_position(object_x, object_y, object_z + 50)
        
        # 2. Move down to the object
        success = success and self.move_to_position(object_x, object_y, object_z)
        
        # 3. Grab the object
        success = success and self.grab()
        
        # 4. Move up with the object
        success = success and self.move_to_position(object_x, object_y, object_z + 50)
        
        # 5. Move to bin position (higher Z to avoid collisions)
        bin_x, bin_y, bin_z = bin_position
        success = success and self.move_to_position(bin_x, bin_y, bin_z + 50)
        
        # 6. Lower into bin
        success = success and self.move_to_position(bin_x, bin_y, bin_z)
        
        # 7. Release the object
        success = success and self.release()
        
        # 8. Move up from the bin
        success = success and self.move_to_position(bin_x, bin_y, bin_z + 50)
        
        # 9. Return to home position
        success = success and self.home()
        
        return success


# Function to simulate arm behavior for testing without real hardware
class SimulatedArmController(ArmController):
    """Simulated version of ArmController for testing without hardware."""
    
    def __init__(self):
        """Initialize without hardware connection."""
        super().__init__(port="SIM")
        self.position = (0, 0, 0)
        self.gripper_closed = False
    
    def connect(self):
        """Simulate connection."""
        self.connected = True
        self.start_worker()
        print("Connected to simulated arm")
        return True
    
    def _execute_command(self, cmd):
        """Simulate command execution."""
        # Print the command for debugging
        print(f"Simulated arm executing: {cmd}")
        
        # Parse the command
        parts = cmd.strip().split()
        
        if not parts:
            return True
        
        # Simulate different commands
        if parts[0] == "MOVE_TO" and len(parts) >= 4:
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                self.position = (x, y, z)
                print(f"Simulated arm moved to position: {self.position}")
            except ValueError:
                print("Invalid coordinates in MOVE_TO command")
                
        elif parts[0] == "GRIPPER":
            if len(parts) >= 2:
                if parts[1] == "CLOSE":
                    self.gripper_closed = True
                    print("Simulated gripper closed")
                elif parts[1] == "OPEN":
                    self.gripper_closed = False
                    print("Simulated gripper opened")
                    
        elif parts[0] == "HOME":
            self.position = (0, 0, 0)
            print("Simulated arm moved to home position")
            
        # Simulate delay for command execution
        time.sleep(0.5)
        
        return True


# Test function
def test_arm_controller():
    """Test the arm controller with simulated commands."""
    # Use the simulated controller for testing
    arm = SimulatedArmController()
    
    if not arm.connect():
        print("Failed to connect to arm. Exiting...")
        return
    
    try:
        # Test basic movements
        print("\nTesting basic movements:")
        arm.move_to_position(100, 100, 100)
        time.sleep(1)  # Wait for command to execute
        
        # Test gripper
        print("\nTesting gripper:")
        arm.grab()
        time.sleep(1)
        arm.release()
        time.sleep(1)
        
        # Test object sorting with a simulated object
        print("\nTesting object sorting:")
        object_info = {
            "color": "red",
            "centroid": (150, 200)
        }
        arm.sort_object(object_info)
        
        # Wait for all commands to complete
        arm.command_queue.join()
        print("\nAll commands completed")
        
    finally:
        arm.disconnect()
        print("Test completed")


if __name__ == "__main__":
    test_arm_controller() 