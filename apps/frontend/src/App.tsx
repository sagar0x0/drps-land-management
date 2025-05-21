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
