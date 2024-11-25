import os.path
import subprocess
import face_recognition
import cv2
import tkinter as tk
import util
from PIL import Image, ImageTk

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        # Creating login button
        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=300)

        # Creating register button
        self.register_button_main_window = util.get_button(self.main_window, 'register', 'gray',
                                                           self.register_new_user, fg="black")
        self.register_button_main_window.place(x=750, y=400)

        # Creating webcam label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        # create directory for our database
        self.db_dir = "./db"
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        unknown_img_path = './.tem.jpg'
        rgb_image = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(unknown_img_path, rgb_image)

        try:
            output = subprocess.check_output(["face_recognition", self.db_dir, unknown_img_path],
                                             stderr=subprocess.STDOUT)
            output = output.decode().strip()
            name = output.split(',')[1][:-3]

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back!', 'Welcome, {}.'.format(name))
        except subprocess.CalledProcessError as e:
            error_msg = e.output.decode()
            print("Error: ", error_msg)
            util.msg_box('Error', f"Failed to recognize face: {error_msg}")
        finally:
            if os.path.exists(unknown_img_path):
                os.remove(unknown_img_path)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        # Creating accept button
        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept',
                                                                      'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        # Creating try again button
        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again',
                                                                         'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        # Creating webcam label
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, "Please, input username:")
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()
    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
