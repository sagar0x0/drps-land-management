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
