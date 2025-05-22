# Next Steps for Your Land Registry Project

Based on your progress and the steps you've already followed, here's a detailed guide on what to do next to get your system fully operational:

## 1. Fix the Frontend Configuration

There are a few issues in your frontend setup that need to be addressed:

### Fix the LandRegistry Import Path

The import path in `blockchain.ts` is incorrect. You need to properly import the ABI from your blockchain project:

```typescript
// Change this line in src/blockchain.ts
import LandRegistry from "../blockchain/artifacts/contracts/LandRegistry.sol/LandRegistry.json";

// To this (use a relative path to your blockchain artifacts)
import LandRegistry from "../../blockchain/artifacts/contracts/LandRegistry.sol/LandRegistry.json";
```

Alternatively, you can copy the ABI file to your frontend project:

```bash
# Create a directory for contract ABIs
mkdir -p apps/frontend/src/contracts

# Copy the ABI file
cp apps/blockchain/artifacts/contracts/LandRegistry.sol/LandRegistry.json apps/frontend/src/contracts/

# Then update the import in blockchain.ts
import LandRegistry from "../contracts/LandRegistry.json";
```


### Fix the File Input in DocumentUpload.tsx

There's a syntax error in your DocumentUpload component:

```typescript
// Change this line
<input type="file" onChange={(e) => setFile(e.target.files[^0])} />

// To this
<input type="file" onChange={(e) => e.target.files && setFile(e.target.files[^0])} />
```


## 2. Set Up App.tsx to Use Your Components

Create or modify `apps/frontend/src/App.tsx`:

```typescript
import React from 'react';
import { ChakraProvider, Container, VStack, Heading, Divider } from '@chakra-ui/react';
import WalletConnector from './components/WalletConnector';
import PropertyRegistry from './components/PropertyRegistry';
import DocumentUpload from './components/DocumentUpload';

function App() {
  return (
    <ChakraProvider>
      <Container maxW="container.md" py={8}>
        <VStack spacing={8} align="stretch">
          <Heading as="h1" size="xl" textAlign="center">Land Registry System</Heading>
          
          <WalletConnector />
          
          <Divider />
          <Heading as="h2" size="lg">Register Property</Heading>
          <PropertyRegistry />
          
          <Divider />
          <Heading as="h2" size="lg">Document Processing</Heading>
          <DocumentUpload />
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App;
```


## 3. Fix the Backend Server

As we discussed earlier, you need to fix the TypeScript errors in your backend server. Here's the corrected version using express-async-handler:

```bash
# Navigate to backend directory
cd apps/backend

# Install required packages if you haven't already
npm install express multer tesseract.js cors
npm install --save-dev typescript ts-node @types/express @types/multer @types/node @types/cors express-async-handler
```

Create a new `server.ts` file:

```typescript
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
```


## 4. Create a Development Workflow

Now that you have all the components, let's set up a workflow to run everything together:

1. Create a root package.json to manage all services:
```bash
# Navigate to project root
cd ~/workspace/github.com/Free_Meals/drps-land-management

# Create or update package.json
cat > package.json << EOF
{
  "name": "drps-land-management",
  "version": "1.0.0",
  "private": true,
  "workspaces": [
    "apps/*"
  ],
  "scripts": {
    "blockchain": "cd apps/blockchain && npx hardhat node",
    "deploy:local": "cd apps/blockchain && npx hardhat run scripts/deploy.js --network localhost",
    "frontend": "cd apps/frontend && npm start",
    "backend": "cd apps/backend && npx ts-node server.ts",
    "dev": "concurrently \"npm run blockchain\" \"npm run frontend\" \"npm run backend\""
  },
  "devDependencies": {
    "concurrently": "^8.2.0"
  }
}
EOF

# Install dependencies
npm install
```


## 5. Start the Development Environment

Now you can start all services together:

```bash
# In one terminal, start the blockchain
npm run blockchain

# In another terminal, deploy the contract
npm run deploy:local

# In a third terminal, start the frontend and backend
npm run frontend
# In a fourth terminal
npm run backend
```

Alternatively, if you installed concurrently, you can run:

```bash
npm run dev
```


## 6. Connect MetaMask to Your Local Blockchain

1. Open MetaMask
2. Add a new network with these settings:
    - Network Name: Hardhat Local
    - New RPC URL: http://127.0.0.1:8545
    - Chain ID: 1337
    - Currency Symbol: ETH
3. Import a test account from Hardhat:
    - Copy a private key from the Hardhat console output
    - In MetaMask, click on your account icon > Import Account
    - Paste the private key and click "Import"

## 7. Test the Complete Flow

1. Open your frontend application (typically at http://localhost:3000)
2. Connect your wallet using the "Connect Wallet" button
3. Register a property by filling in the form and clicking "Register Property"
4. Upload a document to test the OCR functionality

## 8. Next Development Steps

Once you have the basic system working, you can enhance it with:

1. **Property Listing and Viewing**:
    - Create a component to list all registered properties
    - Add a detail view for individual properties
2. **Map Integration**:

```bash
cd apps/frontend
npm install react-leaflet leaflet @types/leaflet
```

3. **Enhanced Document Processing**:
    - Add more sophisticated NLP to extract property details
    - Implement document verification
4. **XDC Network Deployment**:
    - Once everything works locally, deploy to XDC Apothem testnet

```bash
cd apps/blockchain
npx hardhat run scripts/deploy.js --network xdc_testnet
```

5. **Mobile Responsiveness**:
    - Ensure your UI works well on mobile devices

Remember to update your contract address in `blockchain.ts` whenever you deploy to a different network. The current address (`0x5FbDB2315678afecb367f032d93F642f64180aa3`) is specific to your local Hardhat deployment.

This comprehensive approach will get your land registry system up and running with all the core functionality in place.

<div style="text-align: center">⁂</div>

[^1]: paste.txt
