# 🖌️ Neural Style Transfer: Reimagine Images Through AI

Transform your photos into works of art, powered by AI and deep learning! This project brings the power of neural style transfer to your fingertips, letting you blend the content of one image with the artistic style of another. I built this project using PyTorch and PyQt6 as a hands-on way to explore how AI can see, understand, and reimagine images, all through an interactive desktop app.

## 🧠 What is Neural Style Transfer?
Neural Style Transfer is a process that uses deep learning to combine two images, one for content (like a photo of a landscape) and one for style (like a painting by Van Gogh). The result is a new image that keeps the structure of the original photo but adopts the artistic style of the second image. It works by analyzing both images through a neural network and balancing their visual features to generate a blended output.

---

## 💡 Why I Built This

As someone fascinated by the intersection of art and AI, I wanted to explore how deep learning could be used to create something both beautiful and interactive. My project sought to explore neural style transfer, a technique that uses convolutional neural networks (CNNs) to reimagine ordinary images in the styles of famous artworks. I took this project as an opportunity to go beyond theory; translating a research paper into a real, working application helped me strengthen both my technical skills and my ability to build practical AI tools from the ground up.

Through this project, I learned how to work directly with PyTorch to build and customize a deep learning model. I also explored how different parts of the model influence the final results and experimented with how to balance artistic style and image content. Finally, I learned how to package everything into a desktop app that anyone can use which I believe is a valuable step in making machine learning more accessible and user-friendly.

---

## 🧠 How Neural Style Transfer Works

Neural Style Transfer combines two images:

- **Content Image**: The subject (e.g., a photo of a tiger)  
- **Style Image**: The look and texture (e.g., Van Gogh’s *Starry Night*)

By using a pre-trained VGG19 model, my app compares how much the generated image differs from the original content and the chosen style. It does this by calculating two things: content loss (how well the subject is preserved) and style loss (how well the artistic feel is transferred). My model then adjusts the image to balance both, aiming to keep the structure of the original photo while painting it in the chosen style.

> 🧪 I inserted custom loss layers at different points in the VGG19 model to see how each level influenced style and content. This helped me learn how deeper or shallower neural network layers affect the final look, from preserving structure to enhancing texture, and how to fine-tune that balance for better results.

---

## 🧰 Key Features

- 🎨 Upload any content and style image
- ⚖️ Adjust content/style influence with intuitive sliders
- 📷 Progress tracking and real-time stylized output
- 💾 One-click save for your final creation
- 🌓 Beautiful dark-themed UI
- 🖥️ CPU-compatible (no GPU needed)

---

## 🧭 What I Learned

### 🔍 Deep Learning + PyTorch
- Inserted **custom content and style loss layers** directly into a VGG19 model to guide optimization
- Gained a deeper understanding of how different convolutional layers capture content structure vs. artistic texture
- Practiced working with pretrained models, backpropagation, and loss balancing in PyTorch

### 🧪 Experimentation & Tuning
- Explored how loss layer placement affects style transfer outcomes
- Tuned loss weights to control realism vs. abstraction in generated images
- Observed how subtle tweaks in configuration lead to noticeable artistic differences
  
### 🧱 Software Engineering
- Built a modular system with clean separation between model logic and UI
- Designed a responsive UI using PyQt6 that feels smooth and polished
- Packaged the app into an **executable** for easy sharing and use

---

## 🖼️ My Examples

Below are examples of stylized output using different content/style weight settings:
> Note: These examples were generated on my CPU-only setup, which resulted in lower visual quality or less refined details compared to GPU-accelerated results.

<div align="center">
  <img src="https://raw.githubusercontent.com/Akhan521/Neural-Style-Transfer/main/screenshots/content_loss_only.png" width="70%" alt="Content Loss Only" />
  <p><i>Content-focused stylization (minimal style)</i></p>
  <br/>
  <img src="https://raw.githubusercontent.com/Akhan521/Neural-Style-Transfer/main/screenshots/style_loss_only.png" width="70%" alt="Style Loss Only" />
  <p><i>Style-focused stylization (no clear content)</i></p>
  <br/>
  <img src="https://raw.githubusercontent.com/Akhan521/Neural-Style-Transfer/main/screenshots/tiger_as_starry_night.png" width="70%" alt="Tiger in Starry Night Style" />
  <p><i>Balanced stylization (best of both worlds)</i></p>
</div>

---

## 🚀 How to Get Started

### 🖥️ Option 1: Download Executable (No Python Needed)
- [Download for Windows](https://github.com/Akhan521/Neural-Style-Transfer/releases/download/v1.0.1/app.exe)  
- Open the app and start stylizing!

### 🧪 Option 2: Run Locally

#### 0. Open Your Terminal
- On **Windows**: Search for **"Command Prompt"** or **"PowerShell"** in the Start Menu.  
- On **Mac**: Open **Terminal** from Launchpad or Spotlight.  

#### 1. Clone The Repo
```bash
git clone https://github.com/Akhan521/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```
#### 2. Create A Virtual Environment (Optional)
```bash
python -m venv venv
venv\Scripts\activate # On Windows
```
#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 4. Run The App
```bash
python app.py
```
## 🗺️ How to Use It
1. Launch the app
2. Load a content image (the subject)
3. Load a style image (the artwork or texture)
4. Adjust weights if needed using the sliders
5. Click Run Style Transfer
6. View the result and save your image!

## 📚 Acknowledgements
- Inspired by the original NST paper by [Gatys et al](https://arxiv.org/abs/1508.06576).
- Uses a pre-trained VGG19 model from PyTorch
- UI built with PyQt6

## 👨‍💻 My Details
Aamir Khan | ✨ [Portfolio](https://aamir-khans-portfolio.vercel.app/) | 💼 [LinkedIn](https://www.linkedin.com/in/aamir-khan-aak521/) | 💻 [GitHub](https://github.com/Akhan521)

If you enjoyed this project or found it useful, feel free to ⭐ the repo or reach out!
