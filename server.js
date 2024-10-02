import express from 'express';
import multer from 'multer';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;

// Rest of your code remains the same

// Set up multer for handling file uploads
const upload = multer({ dest: 'uploads/' });

// Serve static files from the React app
app.use(express.static(path.join(__dirname, 'dist')));

// API endpoint for Pokemon detection
app.post('/api/detect-pokemon', upload.single('image'), (req, res) => {
    console.log('Received request to /api/detect-pokemon');

    if (!req.file) {
        console.log('No file uploaded');
        return res.status(400).send('No file uploaded.');
    }

    console.log('File received:', req.file.path);

    const pythonProcess = spawn('py', ['autoencoder_script.py', req.file.path]);

    console.log('Python process spawned');

    let result = '';

    pythonProcess.stdout.on('data', (data) => {
        console.log('Received data from Python script:', data.toString());
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error('Python script error:', data.toString());
    });

    pythonProcess.on('close', (code) => {
        console.log('Python process closed with code:', code);
        if (code !== 0) {
            return res.status(500).send('Error processing image');
        }
        try {
            const jsonResult = JSON.parse(result);
            console.log('Parsed result:', jsonResult);
            res.json(jsonResult);
        } catch (error) {
            console.error('Error parsing Python script output:', error);
            res.status(500).send('Error processing image');
        }
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});