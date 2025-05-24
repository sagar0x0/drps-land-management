import os
import json
import re
import asyncio
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File , Form
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from asyncio import Semaphore
# OCR imports
from pdf2image import convert_from_path
import pytesseract
import yaml

# Pydantic Models for API
class DocumentMetadata(BaseModel):
    document_type: str
    certificate_number: str
    execution_date: str
    registration_date: str
    registration_number: str
    registration_book: str
    registration_volume: str
    stamp_duty_paid: float
    registration_fee: float
    document_hash: str

class OwnershipHistory(BaseModel):
    previous_transaction: Dict[str, Any]

class VerificationResult(BaseModel):
    overall_score: float
    authenticity_status: str
    verification_details: Dict[str, Any]
    discrepancies: List[Dict[str, Any]]
    timestamp: str

class CrossAgentRequest(BaseModel):
    function_name: str
    payload: Dict[str, Any]
    agent_id: str
    request_id: Optional[str] = None

class CrossAgentResponse(BaseModel):
    success: bool
    result: Any = None
    error: Optional[str] = None
    request_id: Optional[str] = None

class DocumentVerificationAgent:
    def __init__(self, api_key, max_concurrent_requests=5):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        self.model = "sonar"
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        self.semaphore = Semaphore(max_concurrent_requests)
        
        # Stamp duty rates by state/locality (example rates)
        self.stamp_duty_rates = {
            "delhi": 0.05,  # 5%
            "mumbai": 0.05,  # 5%
            "bangalore": 0.055,  # 5.5%
            "default": 0.05  # 5% default
        }

    async def extract_document_info(self, pdf_path: str):
        """Extract key information from document text asynchronously"""
        try:
            # Convert PDF to image
            images = convert_from_path(pdf_path)

            # Collect OCR results
            document_text = ""
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img)
                document_text += f"Page {i+1}:\n{text.strip()}\n{'-'*40}\n"

            # Load YAML schema
            #try:
            with open("/Users/sagar/Desktop/Hackathon/Perplexity/ai-swarm-agent/data/schema.yaml", "r") as f:
                schema = yaml.safe_load(f)
            # (If you still want to emit it as YAML string, you can dump it back out)
            schema_yaml = yaml.safe_dump(schema, sort_keys=False)
            ''' except FileNotFoundError:
                # Fallback schema if file not found
                schema_yaml = """
                title: Real Estate Transaction Record
                type: object
                required:
                - document_metadata
                - property_details
                - transaction_details
                - parties
                - ownership_history
                """
            '''
            prompt = f"""
            Extract the following information from this land document OCR text and format as JSON:
            
            JSON Schema:
            {schema_yaml}

            Document text:
            {document_text}

            Return ONLY valid JSON with the exact field names from the schema. If a field cannot be determined, set it to null.
            Ensure all required fields are present.

            keep it accurate and conscise
            """

            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a document analysis expert that returns only structured JSON data conforming to the provided schema."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )

            result = response.choices[0].message.content

            # Extract JSON from response
            if "```json" in result:
                start = result.find("```json") + len("```json")
                end = result.find("```", start)
                result = result[start:end].strip()
            elif "{" in result and "}" in result:
                start = result.find("{")
                end = result.rfind("}") + 1
                result = result[start:end]

            return json.loads(result)
            
        except Exception as e:
            return {"error": f"Failed to extract document information: {str(e)}"}
    

    async def verify_certificate_number(self, certificate_number: str, document_type: str) -> Dict[str, Any]:
        """Validate certificate number against government registry using AI verification"""
        
        prompt = f"""
        Verify the authenticity of this certificate number for a real estate document:
        
        Certificate Number: {certificate_number}
        Document Type: {document_type}

       
        Certificate Number Parsing Logic :
        Format Recognition System: For certificates like "IN-DL61365966288224T":

        Country Code Parsing (IN = India)

        Document Type Identification (DL = Delhi, UP = Uttar pradesh) state/union territory short form

        Region Code Extraction (61 = specific region identifier)

        Validation Checksum (T = verification digit)



        
        Check the following:
        1. Does the format match standard government certificate number patterns and Certificate Number Parsing Logic?
        2. Are there any obvious formatting inconsistencies?
        3. Does the certificate number structure align with the document type?
        4. Any red flags in the certificate number pattern?
        
        Return a JSON with:
        - "valid": true/false
        - "confidence": 0.0-1.0
        - "issues": list of any problems found
        - "format_check": description of format validation

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Indian government document verification patterns and certificate number formats and logic."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
            
            result = self._parse_json_response(response.choices[0].message.content)
            
            # Basic format validation as fallback
            if "error" in result:
                basic_check = self._basic_certificate_validation(certificate_number, document_type)
                return basic_check
                
            return result
            
        except Exception as e:
            return {
                "valid": False,
                "confidence": 0.0,
                "issues": [f"Verification failed: {str(e)}"],
                "format_check": "AI verification unavailable"
            }

    def _basic_certificate_validation(self, certificate_number: str, document_type: str) -> Dict[str, Any]:
        """Basic certificate number validation as fallback"""
        if not certificate_number or not document_type:
            return {
                "valid": False,
                "confidence": 0.0,
                "issues": ["Missing certificate number or document type"],
                "format_check": "Basic validation failed"
            }
        
        # Check basic format patterns
        issues = []
        confidence = 0.7
        
        if len(certificate_number) < 10:
            issues.append("Certificate number seems too short")
            confidence -= 0.2
            
        if not re.match(r"^[A-Z0-9\-]+$", certificate_number):
            issues.append("Certificate number contains invalid characters")
            confidence -= 0.3
            
        return {
            "valid": len(issues) == 0,
            "confidence": max(0.0, confidence),
            "issues": issues,
            "format_check": "Basic pattern validation completed"
        }

    async def verify_stamp_duty_calculation(self, transaction_amount: float, stamp_duty_paid: float, property_address: str) -> Dict[str, Any]:
        """Validate stamp duty compliance with local regulations"""
        
        prompt = f"""
        Verify the stamp duty calculation for this real estate transaction:
        
        Transaction Amount: ₹{transaction_amount:,.2f}
        Stamp Duty Paid: ₹{stamp_duty_paid:,.2f}
        Locality: {property_address}
        
        Please check:
        1. What is the applicable stamp duty rate for this locality/state wise in India (search the web)?
        2. Calculate the expected stamp duty amount
        3. Is the paid amount correct within acceptable tolerance?
        4. Any additional considerations for this type of transaction?
        
        Return JSON with:
        - "compliant": true/false
        - "expected_amount": calculated expected stamp duty
        - "deviation_percentage": percentage difference from expected
        - "applicable_rate": rate percentage used
        - "issues": list of any compliance issues

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Indian real estate stamp duty regulations and calculations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
            
            result = self._parse_json_response(response.choices[0].message.content)
            
            # Basic calculation as fallback
            if "error" in result:
                basic_check = self._basic_stamp_duty_calculation(transaction_amount, stamp_duty_paid, property_address)
                return basic_check
                
            return result
            
        except Exception as e:
            return self._basic_stamp_duty_calculation(transaction_amount, stamp_duty_paid, property_address)

    def _basic_stamp_duty_calculation(self, transaction_amount: float, stamp_duty_paid: float, locality: str) -> Dict[str, Any]:
        """Basic stamp duty calculation as fallback"""
        locality_lower = locality.lower()
        
        # Determine applicable rate
        applicable_rate = self.stamp_duty_rates.get("default", 0.05)
        for location, rate in self.stamp_duty_rates.items():
            if location in locality_lower:
                applicable_rate = rate
                break
        
        expected_amount = transaction_amount * applicable_rate
        deviation = ((stamp_duty_paid - expected_amount) / expected_amount) * 100 if expected_amount > 0 else 0
        
        issues = []
        if abs(deviation) > 10:  # More than 10% deviation
            issues.append(f"Stamp duty deviation of {deviation:.1f}% from expected amount")
        
        return {
            "compliant": abs(deviation) <= 10,
            "expected_amount": expected_amount,
            "deviation_percentage": deviation,
            "applicable_rate": applicable_rate * 100,
            "issues": issues
        }

    async def verify_registration_details(self, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate registration completeness and accuracy"""
        
        required_fields = [
            "registration_number", "registration_book", "registration_volume",
            "execution_date", "registration_date", "certificate_number"
        ]
        
        issues = []
        missing_fields = []
        
        # Check required fields
        for field in required_fields:
            if field not in document_metadata or not document_metadata[field]:
                missing_fields.append(field)
        
        if missing_fields:
            issues.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Check date consistency
        try:
            if "execution_date" in document_metadata and "registration_date" in document_metadata:
                exec_date = datetime.fromisoformat(document_metadata["execution_date"])
                reg_date = datetime.fromisoformat(document_metadata["registration_date"])
                
                if reg_date < exec_date:
                    issues.append("Registration date cannot be before execution date")
                    
                # Check if registration is within reasonable time
                days_diff = (reg_date - exec_date).days
                if days_diff > 120:  # More than 4 months
                    issues.append(f"Registration delayed by {days_diff} days from execution")
                    
        except (ValueError, TypeError):
            issues.append("Invalid date format in execution_date or registration_date")
        
        # Verify registration number format
        reg_number = document_metadata.get("registration_number", "")
        if reg_number and not re.match(r"^\d+$", str(reg_number)):
            issues.append("Registration number should be numeric")
        
        confidence = 1.0 - (len(issues) * 0.2)
        
        return {
            "valid": len(issues) == 0,
            "confidence": max(0.0, confidence),
            "issues": issues,
            "missing_fields": missing_fields,
            "verification_status": "complete" if len(issues) == 0 else "incomplete"
        }

    # update this using input form the xdc source 
    async def verify_document_hash_integrity(self, document_hash: str, document_content: str) -> Dict[str, Any]:
        """Validate document hasn't been tampered with using cryptographic verification"""
        
        try:
            # Calculate hash of current document content
            calculated_hash = hashlib.sha256(document_content.encode('utf-8')).hexdigest()
            
            # Remove any prefixes/suffixes from provided hash
            clean_provided_hash = re.sub(r'[^a-fA-F0-9]', '', document_hash.lower())
            clean_calculated_hash = calculated_hash.lower()
            
            integrity_valid = clean_provided_hash == clean_calculated_hash
            
            issues = []
            if not integrity_valid:
                issues.append("Document hash mismatch - possible tampering detected")
            
            if len(clean_provided_hash) not in [32, 40, 64]:  # MD5, SHA1, SHA256 lengths
                issues.append("Invalid hash format - not a standard cryptographic hash")
            
            return {
                "valid": integrity_valid,
                "confidence": 1.0 if integrity_valid else 0.0,
                "provided_hash": document_hash,
                "calculated_hash": calculated_hash,
                "issues": issues,
                "hash_algorithm": "SHA256"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "confidence": 0.0,
                "provided_hash": document_hash,
                "calculated_hash": None,
                "issues": [f"Hash verification failed: {str(e)}"],
                "hash_algorithm": "Unknown"
            }

    async def verify_ownership_chain(self, ownership_history: Dict[str, Any], current_sellers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate legal ownership continuity through chain of title"""
        
        try:
            issues = []
            previous_transaction = ownership_history.get("previous_transaction", {})
            
            if not previous_transaction:
                issues.append("No previous transaction history available")
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "issues": issues,
                    "chain_status": "incomplete"
                }
            
            # Extract previous sellers and current seller names
            previous_sellers = previous_transaction.get("prev_sellers", [])
            current_seller_names = {seller.get("name", "").strip().upper() for seller in current_sellers}
            
            # Normalize previous seller names
            previous_seller_names = {name.strip().upper() for name in previous_sellers if name}


            prev_seller_buyers = previous_transaction.get("prev_buyers", [])
             # Normalize previous seller names
            previous_seller_buyers_names = {name.strip().upper() for name in prev_seller_buyers if name}
            
            # Check if current sellers were buyers in previous transaction
            # In a proper chain, current sellers should match previous buyers
            # For simplification, we check if names are consistent
            
            chain_valid = True
            
            if not previous_seller_names:
                issues.append("Previous transaction seller information missing")
                chain_valid = False

            if not previous_seller_buyers_names:
                issues.append("Previous transaction seller information missing")
                chain_valid = False
            
            if not current_seller_names:
                issues.append("Current seller information missing")
                chain_valid = False
            
            
            # Check for name consistency patterns
            if previous_seller_buyers_names and current_seller_names:
                # Allow for some variation in names (spouse names, etc.)
                name_overlap = any(
                    any(current_name in prev_name or prev_name in current_name 
                        for prev_name in previous_seller_names)
                    for current_name in current_seller_names
                )
                
                if not name_overlap:
                    issues.append("No apparent connection between previous and current sellers")
                    chain_valid = False
            
            # Check transaction date sequence
            prev_date_str = previous_transaction.get("date")
            if prev_date_str:
                try:
                    prev_date = datetime.fromisoformat(prev_date_str)
                    current_date = datetime.now()
                    
                    # Previous transaction should be before current
                    if prev_date >= current_date:
                        issues.append("Previous transaction date is not before current transaction")
                        chain_valid = False
                        
                except ValueError:
                    issues.append("Invalid date format in previous transaction")
                    chain_valid = False
            
            confidence = 1.0 if chain_valid else max(0.0, 1.0 - len(issues) * 0.25)
            
            return {
                "valid": chain_valid,
                "confidence": confidence,
                "issues": issues,
                "chain_status": "valid" if chain_valid else "broken",
                "previous_sellers": list(previous_seller_names),
                "current_sellers": list(current_seller_names)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "confidence": 0.0,
                "issues": [f"Ownership chain verification failed: {str(e)}"],
                "chain_status": "error"
            }

    async def verify_registration_office_jurisdiction(self, property_address: str, registration_office: str) -> Dict[str, Any]:
        """Validate correct registration authority using AI-powered verification"""
        
        prompt = f"""
        Verify if the registration office has proper jurisdiction for this property:
        
        Property Address: {property_address}
        Registration Office: {registration_office}
        
        Check:
        1. Does the registration office typically serve this geographic area (are they near) , also search the web?
        2. Are there any jurisdictional mismatches?
        3. Is this the correct type of registration office for this property type?
        4. Any red flags in office designation or location?
        
        Return JSON with:
        - "valid": true/false
        - "confidence": 0.0-1.0
        - "issues": list of any jurisdictional problems
        - "expected_office": what office should typically handle this area

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Indian property registration jurisdiction and sub-registrar office boundaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
            
            result = self._parse_json_response(response.choices[0].message.content)
                
            # Validate result structure
            if "error" in result:
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "issues": ["Failed to get verification result"],
                    "expected_office": None,
                    "verification_status": "error"
                }
                
            # Ensure required fields are present
            required_fields = ["valid", "confidence", "issues", "expected_office", "verification_status"]
            if not all(field in result for field in required_fields):
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "issues": ["Invalid response format from verification service"],
                    "expected_office": None,
                    "verification_status": "error"
                }
                
            return result
        
        except Exception as e:
            return {
                "valid": False,
                "confidence": 0.0,
                "issues": [f"Verification failed: {str(e)}"],
                "expected_office": None,
                "verification_status": "error"
            }


    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Helper method to extract and parse JSON from response text"""
        if not response_text:
            return {"error": "Empty response text"}
            
        try:
            # First attempt direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Extract JSON if wrapped in code blocks or text
            json_content = response_text
            
            # Try to extract from markdown code blocks
            if "```" in response_text:
                matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if matches:
                    json_content = matches[0]
            
            # Try to extract JSON between curly braces if no code blocks found
            elif "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_content = response_text[start:end]
            
            try:
                return json.loads(json_content.strip())
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text  # Limit response length
                }

    async def comprehensive_document_verification(self, pdf_path: str) -> Dict[str, Any]:
        """Perform comprehensive verification using all verification functions"""
        
        document_data = await self.extract_document_info(pdf_path)

        verification_results = {}
        overall_issues = []
        
        # Extract relevant data sections
        doc_metadata = document_data.get("document_metadata", {})
        property_details = document_data.get("property_details", {})
        transaction_details = document_data.get("transaction_details", {})
        parties = document_data.get("parties", {})
        ownership_history = document_data.get("ownership_history", {})
        
        # 1. Certificate Number Verification
        try:
            cert_result = await self.verify_certificate_number(
                doc_metadata.get("certificate_number", ""),
                doc_metadata.get("document_type", "")
            )
            verification_results["certificate_verification"] = cert_result
            if not cert_result.get("valid", False):
                overall_issues.extend(cert_result.get("issues", []))
        except Exception as e:
            verification_results["certificate_verification"] = {"error": str(e)}
            overall_issues.append(f"Certificate verification failed: {str(e)}")
        
        # 2. Stamp Duty Verification
        try:
            stamp_result = await self.verify_stamp_duty_calculation(
                transaction_details.get("transaction_amount", 0),
                doc_metadata.get("stamp_duty_paid", 0),
                property_details.get("locality", "")
            )
            verification_results["stamp_duty_verification"] = stamp_result
            if not stamp_result.get("compliant", False):
                overall_issues.extend(stamp_result.get("issues", []))
        except Exception as e:
            verification_results["stamp_duty_verification"] = {"error": str(e)}
            overall_issues.append(f"Stamp duty verification failed: {str(e)}")
        
        # 3. Registration Details Verification
        try:
            reg_result = await self.verify_registration_details(doc_metadata)
            verification_results["registration_verification"] = reg_result
            if not reg_result.get("valid", False):
                overall_issues.extend(reg_result.get("issues", []))
        except Exception as e:
            verification_results["registration_verification"] = {"error": str(e)}
            overall_issues.append(f"Registration verification failed: {str(e)}")
        
        # 4. Document Hash Integrity
        try:
            hash_result = await self.verify_document_hash_integrity(
                doc_metadata.get("document_hash", ""),
                json.dumps(document_data, sort_keys=True)
            )
            verification_results["hash_verification"] = hash_result
            if not hash_result.get("valid", False):
                overall_issues.extend(hash_result.get("issues", []))
        except Exception as e:
            verification_results["hash_verification"] = {"error": str(e)}
            overall_issues.append(f"Hash verification failed: {str(e)}")
        
        # 5. Ownership Chain Verification
        try:
            ownership_result = await self.verify_ownership_chain(
                ownership_history,
                parties.get("sellers", [])
            )
            verification_results["ownership_verification"] = ownership_result
            if not ownership_result.get("valid", False):
                overall_issues.extend(ownership_result.get("issues", []))
        except Exception as e:
            verification_results["ownership_verification"] = {"error": str(e)}
            overall_issues.append(f"Ownership verification failed: {str(e)}")
        
        # 6. Registration Office Jurisdiction
        try:
            jurisdiction_result = await self.verify_registration_office_jurisdiction(
                property_details.get("property_address", ""),
                ownership_history.get("previous_transaction", {}).get("registration_office", "")
            )
            verification_results["jurisdiction_verification"] = jurisdiction_result
            if not jurisdiction_result.get("valid", False):
                overall_issues.extend(jurisdiction_result.get("issues", []))
        except Exception as e:
            verification_results["jurisdiction_verification"] = {"error": str(e)}
            overall_issues.append(f"Jurisdiction verification failed: {str(e)}")
        
        # Calculate overall verification score
        valid_checks = sum(1 for result in verification_results.values() 
                          if isinstance(result, dict) and result.get("valid", False))
        total_checks = len([r for r in verification_results.values() if not isinstance(r, dict) or "error" not in r])
        
        overall_score = valid_checks / total_checks if total_checks > 0 else 0.0
        
        # Determine authenticity status
        if overall_score >= self.confidence_thresholds["high"]:
            authenticity_status = "HIGH_CONFIDENCE_AUTHENTIC"
        elif overall_score >= self.confidence_thresholds["medium"]:
            authenticity_status = "MEDIUM_CONFIDENCE_AUTHENTIC"
        elif overall_score >= self.confidence_thresholds["low"]:
            authenticity_status = "LOW_CONFIDENCE_AUTHENTIC"
        else:
            authenticity_status = "SUSPICIOUS_OR_INVALID"
        
        return {
            "overall_score": overall_score,
            "authenticity_status": authenticity_status,
            "verification_details": verification_results,
            "discrepancies": [{"issue": issue, "severity": "high" if "fail" in issue.lower() else "medium"} 
                             for issue in overall_issues],
            "timestamp": datetime.now().isoformat(),
            "total_checks_performed": total_checks,
            "passed_checks": valid_checks
        }

# FastAPI Service Implementation
class DocumentVerificationService:
    def __init__(self):
        self.app = FastAPI(
            title="Document Verification Agent API",
            description="API for comprehensive real estate document verification",
            version="1.0.0"
        )
        
        # Initialize the core agent
        self.agent = DocumentVerificationAgent(
            api_key=os.environ.get("PERPLEXITY_API_KEY")
        )
        
        # Agent endpoints for cross-communication
        self.agent_endpoints = {
            "fraud_detection": "http://localhost:8001",
            "valuation": "http://localhost:8003"
        }
        
        # HTTP client for cross-agent communication
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "agent": "document_verification",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/extract-document-info")
        async def extract_document_info_endpoint(pdf_path: str):
            """Extract structured information from PDF document"""
            try:
                info = await self.agent.extract_document_info(pdf_path)
                return info
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Document extraction failed: {str(e)}")
        
        @self.app.post("/verify-certificate-number")
        async def verify_certificate_number_endpoint(certificate_number: str, document_type: str):
            """Verify certificate number authenticity"""
            try:
                result = await self.agent.verify_certificate_number(certificate_number, document_type)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Certificate verification failed: {str(e)}")

        @self.app.post("/verify-stamp-duty")
        async def verify_stamp_duty_endpoint(transaction_amount: float, stamp_duty_paid: float, locality: str):
            """Verify stamp duty compliance"""
            try:
                result = await self.agent.verify_stamp_duty_calculation(transaction_amount, stamp_duty_paid, locality)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Stamp duty verification failed: {str(e)}")

        @self.app.post("/verify-registration-details")
        async def verify_registration_details_endpoint(document_metadata: DocumentMetadata):
            """Verify registration details completeness"""
            try:
                result = await self.agent.verify_registration_details(document_metadata.dict())
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Registration verification failed: {str(e)}")

        @self.app.post("/verify-document-hash")
        async def verify_document_hash_endpoint(document_hash: str, document_content: str):
            """Verify document hash integrity"""
            try:
                result = await self.agent.verify_document_hash_integrity(document_hash, document_content)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Hash verification failed: {str(e)}")

        @self.app.post("/verify-ownership-chain")
        async def verify_ownership_chain_endpoint(ownership_history: OwnershipHistory, current_sellers: List[Dict[str, Any]]):
            """Verify ownership chain continuity"""
            try:
                result = await self.agent.verify_ownership_chain(ownership_history.dict(), current_sellers)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ownership verification failed: {str(e)}")

        @self.app.post("/verify-registration-office-jurisdiction")
        async def verify_registration_office_jurisdiction_endpoint(property_address: str, registration_office: str):
            """Verify registration office jurisdiction"""
            try:
                result = await self.agent.verify_registration_office_jurisdiction(property_address, registration_office)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Jurisdiction verification failed: {str(e)}")
        
        @self.app.post("/comprehensive-verification", response_model = VerificationResult)
        async def comprehensive_verification_endpoint(pdf_path: Optional[str] = Form(None)):
            """Perform comprehensive document verification"""
            try:
                result = await self.agent.comprehensive_document_verification(pdf_path)
                return VerificationResult(**result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Comprehensive verification failed: {str(e)}")
        

        
        # Cross-agent communication endpoint
        @self.app.post("/cross-agent-call", response_model=CrossAgentResponse)
        async def handle_cross_agent_call(request: CrossAgentRequest):
            """Handle incoming cross-agent function calls"""
            try:
                result = await self.execute_function(request.function_name, request.payload)
                return CrossAgentResponse(
                    success=True,
                    result=result,
                    request_id=request.request_id
                )
            except Exception as e:
                return CrossAgentResponse(
                    success=False,
                    error=str(e),
                    request_id=request.request_id
                )
    
    async def execute_function(self, function_name: str, payload: Dict[str, Any]) -> Any:
        """Handle cross-agent function calls"""
        if function_name == "verify_with_fraud_context":
            return await self.verify_with_fraud_context(**payload)
        elif function_name == "get_document_authenticity":
            return await self.get_document_authenticity(payload["property_id"])
        elif function_name == "comprehensive_verification":
            return await self.agent.comprehensive_document_verification(payload["document_data"])
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def verify_with_fraud_context(self, document_data: Dict[str, Any], fraud_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced verification considering fraud intelligence"""
        if fraud_context and fraud_context.get("enhanced_scrutiny"):
            # Apply stricter thresholds for fraud cases
            original_thresholds = self.agent.confidence_thresholds.copy()
            self.agent.confidence_thresholds["high"] = 0.9
            self.agent.confidence_thresholds["medium"] = 0.75
            
            try:
                result = await self.agent.comprehensive_document_verification(document_data)
                
                # If document shows signs of forgery, alert fraud detection
                if result["overall_score"] < 0.6:
                    await self.report_document_anomaly(
                        property_id=document_data.get("property_details", {}).get("property_id", "unknown"),
                        authenticity_score=result["overall_score"],
                        discrepancies=result.get("discrepancies", [])
                    )
                
                return result
                
            finally:
                # Restore original thresholds
                self.agent.confidence_thresholds = original_thresholds
        else:
            return await self.agent.comprehensive_document_verification(document_data)
    
    async def get_document_authenticity(self, property_id: str) -> Dict[str, Any]:
        """Get document authenticity status for a property"""
        # In a real implementation, this would query a database
        return {
            "property_id": property_id,
            "authenticity_verified": True,
            "confidence": 0.85,
            "last_verification": datetime.now().isoformat(),
            "verification_status": "completed"
        }
    
    async def report_document_anomaly(self, property_id: str, authenticity_score: float, 
                                     discrepancies: List[Dict[str, Any]]):
        """Report document authenticity issues to fraud detection"""
        try:
            await self.call_agent(
                target_agent="fraud_detection",
                function_name="report_document_anomaly",
                property_id=property_id,
                authenticity_score=authenticity_score,
                discrepancies=discrepancies,
                alert_type="document_forgery_suspected"
            )
            print(f"Reported document anomaly for property {property_id}")
        except Exception as e:
            print(f"Failed to report document anomaly: {e}")
    
    async def call_agent(self, target_agent: str, function_name: str, **kwargs) -> Any:
        """Call another agent's function"""
        if target_agent not in self.agent_endpoints:
            raise ValueError(f"Unknown agent: {target_agent}")
            
        url = f"{self.agent_endpoints[target_agent]}/cross-agent-call"
        request_data = CrossAgentRequest(
            function_name=function_name,
            payload=kwargs,
            agent_id="document_verification"
        )
        
        try:
            response = await self.http_client.post(url, json=request_data.dict())
            response.raise_for_status()
            result = CrossAgentResponse(**response.json())
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Agent call failed: {result.error}")
                
        except httpx.RequestError as e:
            raise Exception(f"Network error calling {target_agent}: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error calling {target_agent}: {e.response.status_code}")
    
    def run(self, host="0.0.0.0", port=8002):
        """Run the FastAPI service"""
        uvicorn.run(self.app, host=host, port=port)

# Startup script
async def startup_verification_service():
    """Initialize and start the document verification service"""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")
    
    service = DocumentVerificationService()
    return service

# For testing purposes
if __name__ == "__main__":
    service = DocumentVerificationService()
    service.run()

