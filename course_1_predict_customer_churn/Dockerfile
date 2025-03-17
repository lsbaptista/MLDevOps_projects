# Use the official Python 3.10 base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to leverage caching
COPY requirements_py3.10.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_py3.10.txt

# Install Jupyter and additional tools for notebooks
RUN pip install --no-cache-dir notebook

# Copy the entire project into the container
COPY . .

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Default command is a shell, so you can run both Python scripts and Jupyter
CMD ["bash"]

## Rebuild the container: Open the Command Palette (Ctrl+Shift+P) and select "Dev Containers: Rebuild and Reopen in Container".