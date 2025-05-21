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
