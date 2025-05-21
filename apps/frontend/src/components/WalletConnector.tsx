import { Button } from '@chakra-ui/react';
import { getProvider } from '../blockchain';

export default function WalletConnector() {
  const connectWallet = async () => {
    await window.ethereum.request({ method: 'eth_requestAccounts' });
  };

  return (
    <Button colorScheme="green" onClick={connectWallet}>
      Connect Wallet
    </Button>
  );
}