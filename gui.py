import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class SatelliteGUI:
    def __init__(self, master, countries, train_callback):
        self.master = master
        master.title("Satellite Data Processing")

        self.train_callback = train_callback

        # Attributes for selected directories, conv_size, bin_factor, and model save path
        self.dmsp_folder = None
        self.bm_folder = None
        self.save_path = None  # Save path for the trained model
        self.conv_size = tk.IntVar(master, value=3)  # Default conv_size set to 3
        self.bin_factor = tk.IntVar(master, value=4)  # Default bin_factor set to 4

        # Combobox for country selection
        self.country_var = tk.StringVar(master)
        self.country_combobox = ttk.Combobox(master, textvariable=self.country_var, values=countries, state="readonly")
        self.country_combobox.set("Select a Country")
        self.country_combobox.pack()

        # Entry for setting conv_size
        self.conv_size_label = tk.Label(master, text="Convolution Size:")
        self.conv_size_label.pack()
        self.conv_size_entry = tk.Entry(master, textvariable=self.conv_size)
        self.conv_size_entry.pack()

        # Entry for setting bin_factor
        self.bin_factor_label = tk.Label(master, text="Binning Factor:")
        self.bin_factor_label.pack()
        self.bin_factor_entry = tk.Entry(master, textvariable=self.bin_factor)
        self.bin_factor_entry.pack()

        # Buttons for folder selection, model save location, and model training
        self.dmsp_button = tk.Button(master, text="Select DMSP Data Folder", command=self.select_dmsp_folder)
        self.dmsp_button.pack()

        self.black_marble_button = tk.Button(master, text="Select Black Marble Data Folder", command=self.select_bm_folder)
        self.black_marble_button.pack()

        self.save_path_button = tk.Button(master, text="Select Save Location for Model", command=self.select_save_path)
        self.save_path_button.pack()

        self.train_button = tk.Button(master, text="Train Model", command=self.on_train)
        self.train_button.pack()

    # Methods for folder selection, save path selection, and model training

    def select_dmsp_folder(self):
        self.dmsp_folder = filedialog.askdirectory()
        print("DMSP Data Folder Selected:", self.dmsp_folder)

    def select_bm_folder(self):
        self.bm_folder = filedialog.askdirectory()
        print("Black Marble Data Folder Selected:", self.bm_folder)

    def select_save_path(self):
        self.save_path = filedialog.askdirectory()
        print("Model Save Path Selected:", self.save_path)

    def on_train(self):
        if not self.dmsp_folder or not self.bm_folder or self.country_var.get() == "Select a Country":
            messagebox.showwarning("Warning", "Please select both DMSP and Black Marble data folders and a country.")
            return
        try:
            conv_size_value = self.conv_size.get()  # Retrieve the conv_size value
            bin_factor_value = self.bin_factor.get()  # Retrieve the bin_factor value
        except tk.TclError:
            messagebox.showerror("Error",
                                 "Invalid input. Please enter valid numbers for convolution size and binning factor.")
            return

        # Pass the save_path along with other parameters to the train_callback
        self.train_callback(self.dmsp_folder, self.bm_folder, self.country_var.get(), conv_size_value, bin_factor_value, self.save_path)
