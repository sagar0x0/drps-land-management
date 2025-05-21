require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

// Get private key from environment variables
const PRIVATE_KEY = process.env.PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000000";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    compilers: [
      {
        version: "0.8.28", // Ensure all contracts are compatible with this version
      }
    ],
  },
  networks: {
    // Local development network
    hardhat: {
      chainId: 1337
    },
    // XDC Apothem testnet
    xdc_testnet: {
      url: "https://erpc.apothem.network",
      accounts: [PRIVATE_KEY],
      chainId: 51,
      gasPrice: 20000000000
    },
    // XDC mainnet (for production)
    xdc_mainnet: {
      url: "https://erpc.xinfin.network",
      accounts: [PRIVATE_KEY],
      chainId: 50,
      gasPrice: 20000000000
    }
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};
