FROM python:3.11

WORKDIR /usr/src/app

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm
RUN npm install -g n && n 14 && hash -r && npm install -g npm@6
RUN node --version && npm --version

# Set up Python environment
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set up Node.js environment and copy application
COPY package*.json ./
RUN npm install
COPY . .

# Copy additional files
COPY model.h5 /usr/src/app/
COPY Train /usr/src/app/Train

EXPOSE 3000

CMD ["node", "server.js"]