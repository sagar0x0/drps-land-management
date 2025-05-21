<!-- Creating this file so others can replicate the steps -->

# Dependencies

- node
- npm

```bash
sudo apt install nodejs npm
```

## 2. Blockchain Development Tools

### Install Truffle Suite globally
```bash
npm install -g truffle
```

### Install Ganache for local blockchain development
Download from https://trufflesuite.com/ganache/ or:
```bash
npm install -g ganache-cli
```

## 3. Create Project Structure

```bash
# Create the main project directory
mkdir land-registry-system
cd land-registry-system

# Initialize npm
npm init -y

# Create the basic folder structure
mkdir -p apps/blockchain apps/frontend apps/backend apps/ai libs/common libs/web3-utils libs/ai-models infrastructure/docker infrastructure/ci-cd docs
```

4. Configure Version Control

```bash
# Initialize git repository
git init

# Create .gitignore file
touch .gitignore

# Add common exclusions to .gitignore
echo "node_modules/
.env
.DS_Store
dist/
build/
*.log
__pycache__/
venv/" > .gitignore

## 5. Blockchain Setup

### Set Up Hardhat Project

```bash
# Navigate to blockchain directory
cd apps/blockchain

# Initialize npm for the blockchain directory
npm init -y

# Install Hardhat and essential plugins
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox

# Initialize Hardhat project (choose JavaScript when prompted)
**I have selected JS**
npx hardhat init
```

# Hardhat Project Structure

apps/blockchain/
├── contracts/             # Smart contracts go here
│   └── Lock.sol           # Sample contract
├── scripts/               # Deployment and interaction scripts
│   └── deploy.js          # Sample deployment script
├── test/                  # Test files
│   └── Lock.js            # Sample test
├── hardhat.config.js      # Hardhat configuration
└── package.json           # Project dependencies



# Created initial hardhat.config.js

```JS
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

// Get private key from environment variables
const PRIVATE_KEY = process.env.PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000000";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.20",
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

````

## Install Additional Dependencies

- Install environment variable support and other useful packages:

```bash
# Install dotenv for environment variables
npm install --save-dev dotenv

# Install OpenZeppelin Contracts for standard implementations
npm install @openzeppelin/contracts
```


# Create a Basic Land Registry Contract

Create a new file contracts/LandRegistry.sol:

```text
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

contract LandRegistry is Ownable {
    struct Property {
        string propertyId;
        string location;
        uint256 price;
        address owner;
        uint256 registrationDate;
        bool forSale;
    }
    
    // Mapping from property ID to Property struct
    mapping(string => Property) public properties;
    
    // Array to keep track of all property IDs
    string[] public propertyIds;
    
    // Events
    event PropertyRegistered(string propertyId, address owner, uint256 timestamp);
    event PropertyTransferred(string propertyId, address from, address to, uint256 timestamp);
    event PropertyForSale(string propertyId, uint256 price);
    
    constructor() Ownable(msg.sender) {}
    
    // Function to register a new property
    function registerProperty(
        string memory _propertyId,
        string memory _location,
        uint256 _price
    ) public {
        // Ensure property doesn't already exist
        require(bytes(properties[_propertyId].propertyId).length == 0, "Property already exists");
        
        // Create new property
        Property memory newProperty = Property({
            propertyId: _propertyId,
            location: _location,
            price: _price,
            owner: msg.sender,
            registrationDate: block.timestamp,
            forSale: false
        });
        
        // Store property in mapping and array
        properties[_propertyId] = newProperty;
        propertyIds.push(_propertyId);
        
        // Emit event
        emit PropertyRegistered(_propertyId, msg.sender, block.timestamp);
    }
    
    // Function to transfer ownership
    function transferProperty(string memory _propertyId, address _newOwner) public {
        // Ensure property exists
        require(bytes(properties[_propertyId].propertyId).length > 0, "Property does not exist");
        // Ensure sender is the owner
        require(properties[_propertyId].owner == msg.sender, "Only owner can transfer property");
        
        // Update owner
        address previousOwner = properties[_propertyId].owner;
        properties[_propertyId].owner = _newOwner;
        properties[_propertyId].forSale = false;
        
        // Emit event
        emit PropertyTransferred(_propertyId, previousOwner, _newOwner, block.timestamp);
    }
    
    // Function to list property for sale
    function listPropertyForSale(string memory _propertyId, uint256 _price) public {
        // Ensure property exists
        require(bytes(properties[_propertyId].propertyId).length > 0, "Property does not exist");
        // Ensure sender is the owner
        require(properties[_propertyId].owner == msg.sender, "Only owner can list property");
        
        // Update property
        properties[_propertyId].forSale = true;
        properties[_propertyId].price = _price;
        
        // Emit event
        emit PropertyForSale(_propertyId, _price);
    }
    
    // Function to get total number of properties
    function getPropertyCount() public view returns (uint256) {
        return propertyIds.length;
    }
}
```

## Property Structure 
struct Property {
    string propertyId;         // Unique identifier for the property
    string location;           // Physical location description
    uint256 price;             // Property price
    address owner;             // Ethereum address of current owner
    uint256 registrationDate;  // Timestamp when property was registered
    bool forSale;              // Whether property is listed for sale
}



---


## 10. Create a Deployment Script

Create a new file at `scripts/deploy.js`:

```js
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
```

---

## 11. Create a Test File

Create a new file at `test/LandRegistry.js`:

```js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("LandRegistry", function () {
    let landRegistry;
    let owner;
    let addr1;
    let addr2;

    beforeEach(async function () {
        // Get signers
        [owner, addr1, addr2] = await ethers.getSigners();

        // Deploy the contract
        const LandRegistry = await ethers.getContractFactory("LandRegistry");
        landRegistry = await LandRegistry.deploy();
    });

    describe("Property Registration", function () {
        it("Should register a new property", async function () {
            const propertyId = "LAND123";
            const location = "123 Main St, City";
            const price = ethers.parseEther("1");

            await landRegistry.registerProperty(propertyId, location, price);

            const count = await landRegistry.getPropertyCount();
            expect(count).to.equal(1);

            const property = await landRegistry.properties(propertyId);
            expect(property.owner).to.equal(owner.address);
            expect(property.location).to.equal(location);
            expect(property.price).to.equal(price);
        });

        it("Should not allow registering the same property twice", async function () {
            const propertyId = "LAND123";
            const location = "123 Main St, City";
            const price = ethers.parseEther("1");

            await landRegistry.registerProperty(propertyId, location, price);

            await expect(
                landRegistry.registerProperty(propertyId, location, price)
            ).to.be.revertedWith("Property already exists");
        });
    });

    describe("Property Transfer", function () {
        it("Should transfer property ownership", async function () {
            const propertyId = "LAND123";
            const location = "123 Main St, City";
            const price = ethers.parseEther("1");

            await landRegistry.registerProperty(propertyId, location, price);
            await landRegistry.transferProperty(propertyId, addr1.address);

            const property = await landRegistry.properties(propertyId);
            expect(property.owner).to.equal(addr1.address);
        });

        it("Should not allow non-owners to transfer property", async function () {
            const propertyId = "LAND123";
            const location = "123 Main St, City";
            const price = ethers.parseEther("1");

            await landRegistry.registerProperty(propertyId, location, price);

            await expect(
                landRegistry.connect(addr1).transferProperty(propertyId, addr2.address)
            ).to.be.revertedWith("Only owner can transfer property");
        });
    });
});
```

---

## 12. Create Environment Variables File

Create a `.env` file in the `apps/blockchain` directory:

```env
PRIVATE_KEY=your_private_key_here
```

## 13. Compile and Test
! In this step, we are testing using Hardhat

npx hardhat compile

# Run tests
npx hardhat test

14. Deploy to Local Network

Start a local Hardhat node:

bash
npx hardhat node
In a separate terminal, deploy to the local network:

bash
npx hardhat run scripts/deploy.js --network localhost


15. Deploy to XDC Testnet

To deploy to XDC Testnet (Apothem):

bash
npx hardhat run scripts/deploy.js --network xdc_testnet

