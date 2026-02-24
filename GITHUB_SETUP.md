# GitHub Setup Guide for UCS654 Assignment 6

## Quick Start - Upload to GitHub

Follow these steps to upload your assignment to GitHub and get the submission link:

### Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository settings:
   - **Name**: `UCS654_Assignment_6_Simulation_ML`
   - **Description**: `Data Generation using Pendulum Physics Simulation for Machine Learning`
   - **Visibility**: Public (so instructor can access)
   - **Initialize**: ⚠️ **DO NOT** check any boxes (no README, no .gitignore, no license)
4. Click **"Create repository"**

### Step 2: Upload Files Using Git

#### Option A: Using Git Command Line

Open terminal/PowerShell in the project folder and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit: UCS654 Assignment 6 - Pendulum Simulation ML"

# Add remote repository
git remote add origin https://github.com/aindri1974/UCS654_Assignment_6_102316039.git

# Push to GitHub
git branch -M main
git push -u origin main
```

#### Option B: Using GitHub Desktop (Easier)

1. Download and install [GitHub Desktop](https://desktop.github.com/)
2. Open GitHub Desktop
3. Click **File → Add Local Repository**
4. Select your project folder
5. Click **"Create a repository"**
6. Click **"Publish repository"** to upload to GitHub

#### Option C: Upload via GitHub Web Interface

1. On your new repository page, click **"uploading an existing file"**
2. Drag and drop all files from your project folder
3. Add commit message: "Initial commit: Assignment 6"
4. Click **"Commit changes"**

### Step 3: Verify Upload

1. Go to your repository on GitHub
2. Check that all files are present:
   - ✅ simulation_ml_assignment.ipynb
   - ✅ README.md
   - ✅ requirements.txt
   - ✅ .gitignore
   - ✅ (CSV and PNG files if you ran the notebook)

### Step 4: Enable GitHub Pages (Optional)

If you want to display README as a webpage:
1. Go to **Settings** → **Pages**
2. Source: Deploy from branch → **main** → **/(root)**
3. Click Save

### Step 5: Get Submission Link

Your submission link will be:
```
https://github.com/aindri1974/UCS654_Assignment_6_102316039
```

**This is the link you submit for your assignment!**

---

## Running on Google Colab

### Option 1: Upload Notebook Directly

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `simulation_ml_assignment.ipynb`
4. Run all cells (Runtime → Run all)

### Option 2: Open from GitHub

1. Upload to GitHub first (follow steps above)
2. Go to [Google Colab](https://colab.research.google.com/)
3. Click **File → Open notebook**
4. Select **GitHub** tab
5. Paste your repository URL or search for your username
6. Click on your notebook to open it
7. Run all cells

### Option 3: Add Colab Badge to README

Add this to your README.md to create a direct "Open in Colab" button:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aindri1974/UCS654_Assignment_6_102316039/blob/main/simulation_ml_assignment.ipynb)
```

---

## Tips for Successful Submission

### ✅ Checklist Before Submitting:

- [ ] All code cells run without errors
- [ ] All visualizations display correctly
- [ ] README.md is complete and well-formatted
- [ ] Repository is set to **Public**
- [ ] All required files are uploaded
- [ ] GitHub link is accessible (test in incognito mode)
- [ ] Notebook runs successfully on Google Colab

### 📝 What to Submit:

Submit the GitHub repository URL in this format:
```
https://github.com/aindri1974/UCS654_Assignment_6_102316039
```

### 🎯 Grading Criteria (Typical):

1. **Simulation Implementation** (25%):
   - Correct simulator implementation
   - Appropriate parameter bounds
   - 1000 simulations generated

2. **Code Quality** (20%):
   - Clean, well-documented code
   - Proper structure and organization
   - No errors when running

3. **ML Analysis** (25%):
   - 5-10 models compared
   - Appropriate evaluation metrics
   - Best model identified

4. **Documentation** (20%):
   - Comprehensive README
   - Clear methodology explanation
   - Results properly documented

5. **Visualizations** (10%):
   - Clear, informative plots
   - Proper labels and titles
   - Professional appearance

---

## Troubleshooting

### Common Issues:

**Issue 1: Git not installed**
```bash
# Download and install Git from:
https://git-scm.com/downloads
```

**Issue 2: Authentication required**
```bash
# Use Personal Access Token instead of password
# Generate token at: GitHub Settings → Developer settings → Personal access tokens
```

**Issue 3: Large files warning**
```bash
# If CSV files are too large, add them to .gitignore
echo "*.csv" >> .gitignore
```

**Issue 4: Package installation errors in Colab**
```python
# Restart runtime after installing packages
# Runtime → Restart runtime
```

---

## Need Help?

If you encounter issues:
1. Check the error message carefully
2. Search for the error on Google/Stack Overflow
3. Verify all files are in the correct location
4. Ensure Python 3.7+ is installed
5. Try running in Google Colab (most reliable)

---

## Final Notes

- **Time to Complete**: 30-45 minutes to run all cells
- **Computational Requirements**: Moderate (runs on any modern PC)
- **Storage**: ~5-10 MB for all files
- **Internet Required**: For package installation only

**Good luck with your submission! 🚀**
