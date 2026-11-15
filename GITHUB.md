# Upload to GitHub

Create a new **private** repository named `mri-doctor` at https://github.com/new.
Leave the options to add a README, .gitignore and license unchecked: this project
already contains the relevant files and third-party license notices.

From this project directory, using your own repository URL:

```bash
git remote add origin https://github.com/YOUR_USERNAME/mri-doctor.git
git push -u origin main
```

Authenticate using GitHub's normal sign-in flow or GitHub CLI. Do not paste a
password or access token into the source code or remote URL.

Alternatively, extract the prepared `mri-doctor-github.zip` and upload its
contents through **Add file → Upload files** in your repository. Keep the folder
structure, including the `.gitignore` file. The ZIP itself is a transport archive;
upload its contents to make the source browsable.

The repository excludes model weights, virtual environments, patient studies and
local setup files. To run a clone, follow README.md and download the weights on
that computer. GitHub Pages cannot run the Python backend or local inference.

Reference: [GitHub's instructions for uploading a local repository](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
