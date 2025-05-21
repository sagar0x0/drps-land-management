import express from 'express';
import multer from 'multer';
import { createWorker } from 'tesseract.js';
import asyncHandler from 'express-async-handler';
import cors from 'cors';

const app = express();
app.use(cors());
const upload = multer({ storage: multer.memoryStorage() });

// Initialize worker
let worker: any;
(async () => {
  worker = await createWorker();
  console.log('Tesseract worker initialized');
})();

app.post('/ocr', upload.single('document'), asyncHandler(async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: 'No document uploaded' });
    return;
  }
  
  try {
    const { data: { text } } = await worker.recognize(req.file.buffer);
    res.json({ text });
  } catch (error) {
    console.error('OCR processing error:', error);
    res.status(500).json({ error: 'Failed to process document' });
  }
}));

app.listen(3001, () => console.log('AI Server running on port 3001'));
