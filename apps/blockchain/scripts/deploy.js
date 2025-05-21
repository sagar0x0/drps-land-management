// scripts/deploy.js
const hre = require("hardhat");

async function main() {
  console.log("Deploying LandRegistry contract...");

  const LandRegistry = await hre.ethers.getContractFactory("LandRegistry");
  const landRegistry = await LandRegistry.deploy();

  await landRegistry.waitForDeployment();
  
  const address = await landRegistry.getAddress();
  console.log(`LandRegistry deployed to: ${address}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
