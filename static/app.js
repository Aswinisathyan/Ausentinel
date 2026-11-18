// Upload feedback: show the chosen file, check its size, and show progress.
// The page works without this script; the server repeats every check.

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form.upload");

  if (!form) {
    return;
  }

  const input = form.querySelector('input[type="file"]');
  const nameLabel = form.querySelector("[data-file-name]");
  const errorLabel = form.querySelector("[data-form-error]");
  const button = form.querySelector("[data-submit]");
  const maxBytes = Number(form.dataset.maxMb) * 1024 * 1024;

  const showError = (message) => {
    errorLabel.textContent = message;
    errorLabel.hidden = !message;
  };

  input.addEventListener("change", () => {
    const file = input.files[0];
    nameLabel.textContent = file ? file.name : "";
    showError("");
  });

  form.addEventListener("submit", (event) => {
    const file = input.files[0];

    if (file && file.size > maxBytes) {
      event.preventDefault();
      showError(`This file is larger than the ${form.dataset.maxMb} MB limit.`);
      return;
    }

    button.disabled = true;
    button.textContent = "Analysing...";
  });

  // Re-enable the button if the user comes back with the browser's Back button
  window.addEventListener("pageshow", () => {
    button.disabled = false;
    button.textContent = "Analyse";
  });
});
