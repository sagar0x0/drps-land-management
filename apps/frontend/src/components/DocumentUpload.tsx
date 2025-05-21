import { useState } from 'react';
import { Button, Box } from '@chakra-ui/react';

export default function DocumentUpload() {
  const [file, setFile] = useState<File>();

  const processDocument = async () => {
    const formData = new FormData();
    formData.append('document', file);
    
    const response = await fetch('http://localhost:3001/ocr', {
      method: 'POST',
      body: formData
    });
    
    const { text } = await response.json();
    console.log("Extracted Text:", text);
  };

  return (
    <Box p={4}>
      <input type="file" onChange={(e) => e.target.files && setFile(e.target.files[0])} />
      <Button mt={2} onClick={processDocument}>Process Document</Button>
    </Box>
  );
}