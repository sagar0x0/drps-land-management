import { ethers } from "ethers";
import LandRegistry from "apps/blockchain/artifacts/contracts/LandRegistry.sol/LandRegistry.json";

const contractAddress = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; 

export const getProvider = () => {
  if (typeof window.ethereum !== 'undefined') {
    return new ethers.BrowserProvider(window.ethereum);
  }
  return ethers.getDefaultProvider();
};

export const getContract = async () => {
  const provider = getProvider();
  const signer = await provider.getSigner();
  return new ethers.Contract(contractAddress, LandRegistry.abi, signer);
};