const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const app = express();

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Enable CORS for all origins
app.use(cors());

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Endpoint for fetching JSON files
app.get('/fetch-file', async (req, res) => {
    const dataDir = path.join(__dirname, '..', 'data');
    try {
        const files = await fs.promises.readdir(dataDir);
        const jsonFiles = files.filter(file => file.endsWith('_edit.json'));

        if (jsonFiles.length === 0) {
            return res.status(404).send("No JSON files found.");
        }

        const allFileData = await Promise.all(jsonFiles.map(async (file) => {
            const filePath = path.join(dataDir, file);
            const fileContent = await fs.promises.readFile(filePath, 'utf-8');
            return {
                fileName: file.replace('_edit.json', ''),
                data: JSON.parse(fileContent)
            };
        }));

        // Extract unique segment options
        const segmentOptions = [...new Set(allFileData.flatMap(file => file.data.map(o => o.Segment.toString())))];

        res.json({ allFileData, segmentOptions });
    } catch (err) {
        console.error("Error fetching data: ", err);
        res.status(500).send("Error fetching data.");
    }
});

// Endpoint for exporting JSON file
app.post('/export', async (req, res) => {
    const { fileName, data } = req.body;
    const filePath = path.join(__dirname, '..', 'export', `${fileName}_edited.json`);

    try {
        const { addModeCoordinates, deleteModeCoordinates } = data;

        // Combining both arrays into one for the export
        const combinedCoordinates = [...addModeCoordinates, ...deleteModeCoordinates];

        // Convert JSON to string
        const jsonContent = JSON.stringify(combinedCoordinates, null, 2);

        // Write to file
        await fs.promises.writeFile(filePath, jsonContent);
        console.log('File exported successfully:', fileName);
        res.status(200).send('File exported successfully.');
    } catch (err) {
        console.error('Error exporting file:', err);
        res.status(500).send('Error exporting file.');
    }
});

// Start the server
const port = process.env.PORT || 3001;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});