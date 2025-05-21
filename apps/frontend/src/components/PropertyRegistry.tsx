import { useEffect, useState } from 'react';
import { Button, Input, Box } from '@chakra-ui/react';
import { getContract } from '../blockchain';

export default function PropertyRegistry() {
  const [propertyId, setPropertyId] = useState("");
  const [location, setLocation] = useState("");

  const registerProperty = async () => {
    const contract = await getContract();
    const tx = await contract.registerProperty(propertyId, location, 0);
    await tx.wait();
    alert("Property registered!");
  };

  return (
    <Box p={4} borderWidth={1} borderRadius="lg">
      <Input placeholder="Property ID" onChange={(e) => setPropertyId(e.target.value)} />
      <Input placeholder="Location" onChange={(e) => setLocation(e.target.value)} mt={2} />
      <Button colorScheme="blue" mt={4} onClick={registerProperty}>
        Register Property
      </Button>
    </Box>
  );
}