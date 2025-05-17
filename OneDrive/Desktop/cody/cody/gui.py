import tkinter as tk
import math
import random

class AssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cody - AI Assistant")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#0a0a0a')
        
        # Colors and settings
        self.bg_color = '#0a0a0a'
        self.glow_color = '#4dabf7'
        self.active_glow = '#74c0fc'
        self.wave_color = '#5c7cfa'
        
        # Main Canvas
        self.canvas = tk.Canvas(root, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status Text
        self.status_text = self.canvas.create_text(
            self.root.winfo_screenwidth()//2,
            self.root.winfo_screenheight() - 150,
            text="Say 'Hello' to wake me up...",
            font=('Segoe UI', 20),
            fill='white',
            tags='status'
        )
        
        # Pulse Animation Variables
        self.pulse_radius = 0
        self.pulse_max = 150
        self.pulse_speed = 2
        self.is_listening = False
        
        # Microphone Icon
        self.mic = self.canvas.create_text(
            self.root.winfo_screenwidth()//2,
            self.root.winfo_screenheight()//2,
            text="🎤",
            font=('Segoe UI', 72),
            fill='white',
            tags='mic'
        )
        
        # Speech Waveform
        self.wave_points = []
        self.wave_amplitude = 30
        self._create_waveform()
        
        # Close button
        self.close_btn = tk.Button(root, text="X", command=root.destroy,
                                     font=('Arial', 14), fg='white', bg='#e03131',
                                     borderwidth=0, relief='flat')
        self.close_btn.place(relx=0.98, rely=0.02, anchor='ne')
        
        # Start animations
        self._animate_pulse()
        self._animate_waveform()
    
    def run(self):
        speak("Cody is online. Say 'Hello' to wake me up.")
        while True:
            now = time.time()

            # Check if we're still in the 2-minute window after last wake-up
            if now - self.last_wake_time < 120:
                self.gui.update_status("Listening (no wake word needed)...")
                self.gui.start_listening_animation()
                speak("How can I help?")
                cmd = listen() or ""
                self.gui.update_status(f"Command: {cmd}")
                self.handle(cmd)
                self.gui.update_status("Ready for next command...")
                self.gui.stop_listening_animation()
            else:
                # Wait for the wake word
                text = listen()
                if text and WAKE_WORD in text:
                    self.last_wake_time = time.time()  # Update last wake-up time
                    self.gui.update_status("Listening...")
                    self.gui.start_listening_animation()
                    speak("How can I help?")
                    cmd = listen() or ""
                    self.gui.update_status(f"Command: {cmd}")
                    self.handle(cmd)
                    self.gui.update_status("Say 'Hello' to wake me up...")
                    self.gui.stop_listening_animation()

    def _animate_pulse(self):
        cx = self.root.winfo_screenwidth()//2
        cy = self.root.winfo_screenheight()//2
        self.pulse_radius += self.pulse_speed
        glow_color = self.active_glow if self.is_listening else self.glow_color
        self.canvas.delete('pulse')
        self.canvas.create_oval(
            cx - self.pulse_radius, cy - self.pulse_radius,
            cx + self.pulse_radius, cy + self.pulse_radius,
            outline=glow_color,
            width=3,
            tags='pulse',
            stipple='gray50'
        )
        if self.pulse_radius > self.pulse_max:
            self.pulse_radius = 0
        self.root.after(20, self._animate_pulse)

    def _create_waveform(self):
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        for i in range(100):
            x = i * (width / 100)
            y = height - 100 + random.randint(-self.wave_amplitude, self.wave_amplitude)
            self.wave_points.append((x, y))
        self.canvas.create_line(
            *self.wave_points,
            fill=self.wave_color,
            width=3,
            tags='waveform',
            smooth=True
        )

    def _animate_waveform(self):
        new_points = []
        max_y = self.root.winfo_screenheight() - 100
        for x, y in self.wave_points:
            new_y = y + random.randint(-5, 5)
            new_y = min(max(new_y, max_y - self.wave_amplitude), max_y + self.wave_amplitude)
            new_points.append((x, new_y))
        self.wave_points = new_points
        self.canvas.delete('waveform')
        self.canvas.create_line(
            *self.wave_points,
            fill=self.wave_color,
            width=3,
            tags='waveform',
            smooth=True
        )
        self.root.after(50, self._animate_waveform)

    def update_status(self, text):
        self.canvas.itemconfig(self.status_text, text=text)

    def start_listening_animation(self):
        self.is_listening = True
        self.canvas.itemconfig('mic', fill=self.active_glow)
        self.pulse_speed = 4

    def stop_listening_animation(self):
        self.is_listening = False
        self.canvas.itemconfig('mic', fill='white')
        self.pulse_speed = 2