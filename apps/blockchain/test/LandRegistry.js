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
