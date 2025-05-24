import os
import json
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks , Form
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from asyncio import Semaphore
import numpy as np
# OCR imports
from pdf2image import convert_from_path
import pytesseract
import yaml

# Pydantic Models for API
class TransactionRecord(BaseModel):
    transaction_amount: float
    total_deel_sum: float
    transaction_details: Optional[Dict[str, Any]] = None
    property_details: Optional[Dict[str, Any]] = None

class PaymentMethod(BaseModel):
    amount: float
    method: str
    utr_number: str
    date: str
    bank: str

class OwnershipTransfer(BaseModel):
    date: str
    seller_id: str
    buyer_id: str
    transaction_amount: Optional[float] = None

class Party(BaseModel):
    name: str
    address: str
    spouse: Optional[str] = None
    share: Optional[str] = None

class Parties(BaseModel):
    sellers: List[Party]
    buyer: Party

class TransactionDetails(BaseModel):
    transaction_amount: float
    circle_rate_per_sqm: float
    area_sq_meters: float
    construction_rate_per_sqm: Optional[float] = None

class FraudAnalysisResult(BaseModel):
    suspicious: bool
    risk_score: float
    details: Dict[str, Any]
    analysis_type: str
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

class PropertyFraudDetectionAgent:
    def __init__(self, api_key, max_concurrent_requests=5):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        self.model = "sonar"
        self.semaphore = Semaphore(max_concurrent_requests)
        
        self.risk_thresholds = {
            "high": 0.75,
            "medium": 0.5,
            "low": 0.3
        }
        
        # Transaction pattern cache for analysis
        self.transaction_history_cache = {}
        self.property_risk_cache = {}
        self.alert_history = {}


    
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

    async def analyze_price_deviation(self, transaction_record: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant price deviations from calculated costs"""
        transaction_amount = transaction_record.get('transaction_amount')
        total_deel_sum = transaction_record.get('total_deel_sum')

        
        if transaction_amount is None or total_deel_sum is None:
            return {
                'suspicious': False, 
                'risk_score': 0.0,
                'reason': 'Missing transaction or calculated cost data',
                'analysis_type': 'price_deviation'
            }
        
        if total_deel_sum <= 0:
            return {
                'suspicious': True,
                'risk_score': 0.8,
                'reason': 'Invalid calculated cost (zero or negative)',
                'analysis_type': 'price_deviation'
            }
        
        deviation = abs(transaction_amount - total_deel_sum) / total_deel_sum
        
        # Risk scoring based on deviation magnitude
        if deviation > 0.5:  # 50% deviation
            risk_score = 0.9
            suspicious = True
        elif deviation > 0.3:  # 30% deviation
            risk_score = 0.7
            suspicious = True
        elif deviation > 0.15:  # 15% deviation
            risk_score = 0.4
            suspicious = False
        else:
            risk_score = 0.1
            suspicious = False
        
        # Enhance analysis with AI insights for high-risk cases
        ai_insights = {}
        if risk_score >= 0.7:
            ai_insights = await self._get_ai_price_analysis(transaction_amount, total_deel_sum, deviation)
        
        return {
            'suspicious': suspicious,
            'risk_score': risk_score,
            'deviation': deviation,
            'deviation_percentage': deviation * 100,
            'transaction_amount': transaction_amount,
            'calculated_cost': total_deel_sum,
            'ai_insights': ai_insights,
            'analysis_type': 'price_deviation'
        }
    


    async def analyze_payment_concentration(self, payment_methods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect suspicious payment patterns"""
        if not payment_methods or len(payment_methods) == 0:
            return {
                'suspicious': False, 
                'risk_score': 0.0,
                'reason': 'No payment methods provided',
                'analysis_type': 'payment_concentration'
            }
        
        # Count payments per date and bank
        payments_by_date_bank = defaultdict(float)
        payments_by_date = defaultdict(float)
        payments_by_bank = defaultdict(float)
        
        for payment in payment_methods:
            date = payment.get('date')
            bank = payment.get('bank')
            amount = payment.get('amount', 0)
            
            payments_by_date_bank[(date, bank)] += amount
            payments_by_date[date] += amount
            payments_by_bank[bank] += amount
        
        total_amount = sum(p.get('amount', 0) for p in payment_methods)
        suspicious = False
        risk_score = 0.0
        suspicious_details = []
        
        # Check for concentration on same day and bank
        for (date, bank), amount in payments_by_date_bank.items():
            concentration_ratio = amount / total_amount if total_amount > 0 else 0
            
            if concentration_ratio > 0.8:  # 80% concentration
                suspicious = True
                risk_score = max(risk_score, 0.9)
                suspicious_details.append({
                    'type': 'high_concentration_same_day_bank',
                    'date': date, 
                    'bank': bank, 
                    'amount': amount,
                    'concentration_ratio': concentration_ratio
                })
            elif concentration_ratio > 0.5:  # 50% concentration
                suspicious = True
                risk_score = max(risk_score, 0.6)
                suspicious_details.append({
                    'type': 'moderate_concentration_same_day_bank',
                    'date': date, 
                    'bank': bank, 
                    'amount': amount,
                    'concentration_ratio': concentration_ratio
                })
        
        # Check for same-day payment concentration (any bank)
        for date, amount in payments_by_date.items():
            if amount / total_amount > 0.9:  # 90% on same day
                suspicious = True
                risk_score = max(risk_score, 0.7)
                suspicious_details.append({
                    'type': 'same_day_payment_concentration',
                    'date': date,
                    'amount': amount,
                    'concentration_ratio': amount / total_amount
                })
        
        # Check for unusual payment methods or patterns (more than 20% cash suspicion)
        payment_methods_used = set(p.get('method', '').upper() for p in payment_methods)
        if 'CASH' in payment_methods_used:
            cash_amount = sum(p.get('amount', 0) for p in payment_methods if p.get('method', '').upper() == 'CASH')
            if cash_amount / total_amount > 0.2:  # More than 20% cash
                suspicious = True
                risk_score = max(risk_score, 0.6)
                suspicious_details.append({
                    'type': 'high_cash_component',
                    'cash_amount': cash_amount,
                    'cash_ratio': cash_amount / total_amount
                })
        
        # Enhance with AI analysis for complex patterns
        ai_insights = {}
        if risk_score >= 0.6:
            ai_insights = await self._get_ai_payment_analysis(payment_methods, suspicious_details)
        
        return {
            'suspicious': suspicious,
            'risk_score': risk_score,
            'details': suspicious_details,
            'payment_summary': {
                'total_amount': total_amount,
                'number_of_payments': len(payment_methods),
                'unique_banks': len(payments_by_bank),
                'unique_dates': len(payments_by_date),
                'payment_methods': list(payment_methods_used)
            },
            'ai_insights': ai_insights,
            'analysis_type': 'payment_concentration'
        }

    async def detect_rapid_ownership_transfers(self, property_id: str, ownership_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify flip fraud patterns"""
        if not ownership_history or len(ownership_history) < 1:
            return {
                'suspicious': False, 
                'risk_score': 0.0,
                'reason': 'Insufficient ownership history (need at least 1 transactions)',
                'analysis_type': 'rapid_ownership_transfers'
            }
        
        # Sort transactions by date ascending
        try:
            transactions = sorted(ownership_history, key=lambda x: datetime.fromisoformat(x.get('date', '1900-01-01')))
        except ValueError:
            return {
                'suspicious': False,
                'risk_score': 0.0,
                'reason': 'Invalid date format in ownership history',
                'analysis_type': 'rapid_ownership_transfers'
            }
        
        suspicious = False
        risk_score = 0.0
        suspicious_periods = []
        
        for i in range(1, len(transactions)):
            try:
                prev_date = datetime.fromisoformat(transactions[i-1].get('date'))
                curr_date = datetime.fromisoformat(transactions[i].get('date'))
                delta_days = (curr_date - prev_date).days
                
                severity = "LOW"  # Define with a default value
                if len(transactions) != 1:
                    # Calculate risk based on holding period
                    if delta_days < 30:  # Less than 1 month
                        risk_score = max(risk_score, 0.95)
                        suspicious = True
                        severity = "CRITICAL"
                    elif delta_days < 90:  # Less than 3 months
                        risk_score = max(risk_score, 0.8)
                        suspicious = True
                        severity = "HIGH"
                    elif delta_days < 180:  # Less than 6 months
                        risk_score = max(risk_score, 0.6)
                        suspicious = True
                        severity = "MEDIUM"
                    else:
                        continue  # Assign a default value
                
                # Calculate potential profit/loss if amounts available
                profit_analysis = {}

                if len(transactions) != 1:
                    if 'transaction_amount' in transactions[i-1] and 'transaction_amount' in transactions[i]:
                        prev_amount = transactions[i-1].get('transaction_amount', 0)
                        curr_amount = transactions[i].get('transaction_amount', 0)
                        if prev_amount > 0:
                            profit_percentage = ((curr_amount - prev_amount) / prev_amount) * 100
                            profit_analysis = {
                                'previous_amount': prev_amount,
                                'current_amount': curr_amount,
                                'profit_percentage': profit_percentage,
                                'absolute_profit': curr_amount - prev_amount
                            }
                            
                            # Flag unrealistic appreciation
                            if profit_percentage > 50 and delta_days < 180:  # 50% profit in less than 6 months
                                risk_score = max(risk_score, 0.9)
                
                suspicious_periods.append({
                    'from_date': transactions[i-1].get('date'),
                    'to_date': transactions[i].get('date'),
                    'days': delta_days,
                    'severity': severity,
                    'from_seller': transactions[i-1].get('seller_id'),
                    'to_seller': transactions[i].get('seller_id'),
                    'profit_analysis': profit_analysis
                })
                
            except Exception as e:
                continue
        
        # Enhance with AI analysis for pattern recognition
        ai_insights = {}
        if suspicious and risk_score >= 0.7:
            ai_insights = await self._get_ai_flip_analysis(property_id, suspicious_periods)
        
        return {
            'suspicious': suspicious,
            'risk_score': risk_score,
            'periods': suspicious_periods,
            'property_id': property_id,
            'total_transfers': len(transactions),
            'rapid_transfers_count': len(suspicious_periods),
            'ai_insights': ai_insights,
            'analysis_type': 'rapid_ownership_transfers'
        }

    async def verify_party_identity_consistency(self, parties: Dict[str, Any]) -> Dict[str, Any]:
        """Check for identity-based fraud indicators"""
        if not parties:
            return {
                'suspicious': False, 
                'risk_score': 0.0,
                'reason': 'No party information provided',
                'analysis_type': 'party_identity_consistency'
            }
        
        sellers = parties.get('sellers', [])
        buyer = parties.get('buyer', {})
        
        suspicious = False
        risk_score = 0.0
        issues = []
        
        # Check sellers
        seller_names = set()
        seller_addresses = set()
        
        for seller in sellers:
            name = seller.get('name', '').strip()
            address = seller.get('address', '').strip()
            
            if not name:
                suspicious = True
                risk_score = max(risk_score, 0.7)
                issues.append({'type': 'missing_seller_name', 'seller': seller})
                continue
            
            # Check for duplicate names
            name_normalized = name.lower().replace('.', '').replace(',', '')
            if name_normalized in seller_names:
                suspicious = True
                risk_score = max(risk_score, 0.8)
                issues.append({'type': 'duplicate_seller_name', 'name': name})
            seller_names.add(name_normalized)
            
            # Check address validity
            if not address or len(address) < 10:
                suspicious = True
                risk_score = max(risk_score, 0.6)
                issues.append({'type': 'incomplete_seller_address', 'name': name, 'address': address})
            elif 'unknown' in address.lower() or 'na' in address.lower():
                suspicious = True
                risk_score = max(risk_score, 0.8)
                issues.append({'type': 'suspicious_seller_address', 'name': name, 'address': address})
            
            # Check for address reuse
            address_normalized = address.lower().strip()
            if address_normalized in seller_addresses and len(address_normalized) > 10:
                suspicious = True
                risk_score = max(risk_score, 0.7)
                issues.append({'type': 'duplicate_seller_address', 'address': address})
            seller_addresses.add(address_normalized)
        
        # Check buyer information
        buyer_name = buyer.get('name', '').strip()
        buyer_address = buyer.get('address', '').strip()
        
        if not buyer_name:
            suspicious = True
            risk_score = max(risk_score, 0.8)
            issues.append({'type': 'missing_buyer_name'})
        
        if not buyer_address or len(buyer_address) < 10:
            suspicious = True
            risk_score = max(risk_score, 0.6)
            issues.append({'type': 'incomplete_buyer_address', 'address': buyer_address})
        elif 'unknown' in buyer_address.lower() or 'na' in buyer_address.lower():
            suspicious = True
            risk_score = max(risk_score, 0.8)
            issues.append({'type': 'suspicious_buyer_address', 'address': buyer_address})
        
        # Check for buyer-seller name conflicts
        buyer_name_normalized = buyer_name.lower().replace('.', '').replace(',', '')
        if buyer_name_normalized in seller_names:
            suspicious = True
            risk_score = max(risk_score, 0.9)
            issues.append({'type': 'buyer_seller_name_conflict', 'name': buyer_name})
        
        # Check contact information patterns
        buyer_mobile = buyer.get('mobile', '')
        if buyer_mobile:
            if len(buyer_mobile) != 10 or not buyer_mobile.isdigit():
                suspicious = True
                risk_score = max(risk_score, 0.4)
                issues.append({'type': 'invalid_mobile_format', 'mobile': buyer_mobile})
        
        # Enhance with AI analysis for sophisticated identity fraud
        ai_insights = {}
        if risk_score >= 0.6:
            ai_insights = await self._get_ai_identity_analysis(parties, issues)
        
        return {
            'suspicious': suspicious,
            'risk_score': risk_score,
            'issues': issues,
            'party_summary': {
                'sellers_count': len(sellers),
                'unique_seller_names': len(seller_names),
                'unique_seller_addresses': len(seller_addresses),
                'buyer_name': buyer_name,
                'buyer_contact_available': bool(buyer.get('mobile') or buyer.get('email'))
            },
            'ai_insights': ai_insights,
            'analysis_type': 'party_identity_consistency'
        }

    async def analyze_undervaluation_fraud(self, transaction_details: Dict[str, Any]) -> Dict[str, Any]:
        """Detect systematic undervaluation for tax evasion"""
        transaction_amount = transaction_details.get('transaction_amount')
        circle_rate_per_sqm = transaction_details.get('circle_rate_per_sqm')
        area_sq_meters = transaction_details.get('area_sq_meters')
        
        if None in (transaction_amount, circle_rate_per_sqm, area_sq_meters):
            return {
                'suspicious': False, 
                'risk_score': 0.0,
                'reason': 'Missing required data: transaction_amount, circle_rate_per_sqm, or area_sq_meters',
                'analysis_type': 'undervaluation_fraud'
            }
        
        if circle_rate_per_sqm <= 0 or area_sq_meters <= 0:
            return {
                'suspicious': True,
                'risk_score': 0.8,
                'reason': 'Invalid circle rate or area (zero or negative values)',
                'analysis_type': 'undervaluation_fraud'
            }
        
        # Calculate expected minimum value based on circle rate
        expected_land_value = circle_rate_per_sqm * area_sq_meters
        
        # Add construction value if available
        construction_rate_per_sqm = transaction_details.get('construction_rate_per_sqm', 0)
        expected_construction_value = construction_rate_per_sqm * area_sq_meters if construction_rate_per_sqm else 0
        
        total_expected_value = expected_land_value + expected_construction_value
        
        # Calculate deviation
        if total_expected_value > 0:
            deviation = (total_expected_value - transaction_amount) / total_expected_value
        else:
            deviation = 0
        
        # Risk assessment based on undervaluation
        suspicious = False
        risk_score = 0.0
        
        if deviation > 0.5:  # More than 50% undervaluation
            suspicious = True
            risk_score = 0.95
            severity = "CRITICAL"
        elif deviation > 0.3:  # More than 30% undervaluation
            suspicious = True
            risk_score = 0.8
            severity = "HIGH"
        elif deviation > 0.15:  # More than 15% undervaluation
            suspicious = True
            risk_score = 0.6
            severity = "MEDIUM"
        elif deviation > 0:  # Any undervaluation
            risk_score = 0.3
            severity = "LOW"
        else:  # Overvaluation or exact match
            risk_score = 0.1
            severity = "NONE"
        
        # Calculate tax evasion potential

        stamp_duty_rate = transaction_details.get('stamp_duty_rate', .05)  # Assume 5% stamp duty
        potential_tax_saved = deviation * total_expected_value * stamp_duty_rate if deviation > 0 else 0
        
        # Enhance with AI analysis for market context
        ai_insights = {}
        if risk_score >= 0.6:
            ai_insights = await self._get_ai_valuation_analysis(transaction_details, deviation, total_expected_value)
        
        return {
            'suspicious': suspicious,
            'risk_score': risk_score,
            'deviation': deviation,
            'deviation_percentage': deviation * 100,
            'severity': severity,
            'valuation_breakdown': {
                'transaction_amount': transaction_amount,
                'expected_land_value': expected_land_value,
                'expected_construction_value': expected_construction_value,
                'total_expected_value': total_expected_value,
                'undervaluation_amount': max(0, total_expected_value - transaction_amount)
            },
            'tax_implications': {
                'potential_stamp_duty_saved': potential_tax_saved,
                'estimated_stamp_duty_rate': stamp_duty_rate
            },
            'ai_insights': ai_insights,
            'analysis_type': 'undervaluation_fraud'
        }

    # AI Enhancement Methods
    async def _get_ai_price_analysis(self, transaction_amount: float, calculated_cost: float, deviation: float) -> Dict[str, Any]:
        """Get AI insights on price deviation patterns"""
        prompt = f"""
        Analyze this real estate price deviation for fraud indicators:
        
        Transaction Amount: ₹{transaction_amount:,.2f}
        Calculated Cost: ₹{calculated_cost:,.2f}
        Deviation: {deviation:.2%}
        
        Provide insights on:
        1. Common fraud patterns that show this type of price deviation
        2. Legitimate reasons for such deviations
        3. Additional red flags to investigate
        4. Risk assessment for money laundering or tax evasion
        
        Return JSON with keys: fraud_patterns, legitimate_reasons, red_flags, risk_assessment

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in real estate fraud detection and financial crime analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _get_ai_payment_analysis(self, payment_methods: List[Dict[str, Any]], suspicious_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get AI insights on payment concentration patterns"""
        prompt = f"""
        Analyze these payment patterns for fraud indicators:
        
        Payment Methods: {json.dumps(payment_methods, indent=2)}
        Detected Issues: {json.dumps(suspicious_details, indent=2)}
        
        Evaluate:
        1. Money laundering risk based on payment patterns
        2. Structuring or smurfing indicators
        3. Source of funds concerns
        4. Banking relationship patterns
        
        Return JSON with keys: laundering_risk, structuring_indicators, source_concerns, banking_patterns

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in anti-money laundering and payment fraud detection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _get_ai_flip_analysis(self, property_id: str, suspicious_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get AI insights on property flipping patterns"""
        prompt = f"""
        Analyze this property flipping pattern for fraud indicators:
        
        Property ID: {property_id}
        Suspicious Periods: {json.dumps(suspicious_periods, indent=2)}
        
        Assess:
        1. Organized fraud scheme indicators
        2. Market manipulation potential
        3. Coordinated buyer/seller networks
        4. Artificial value inflation techniques
        
        Return JSON with keys: scheme_indicators, manipulation_risk, network_analysis, inflation_techniques

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in real estate fraud schemes and market manipulation detection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _get_ai_identity_analysis(self, parties: Dict[str, Any], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get AI insights on identity fraud patterns"""
        prompt = f"""
        Analyze these party identity patterns for fraud indicators:
        
        Parties: {json.dumps(parties, indent=2)}
        Detected Issues: {json.dumps(issues, indent=2)}
        
        Evaluate:
        1. Synthetic identity fraud indicators
        2. Shell entity patterns
        3. Nominee arrangement signs
        4. Identity theft red flags
        
        Return JSON with keys: synthetic_identity_risk, shell_entity_indicators, nominee_patterns, theft_indicators

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in identity fraud and financial crime detection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _get_ai_valuation_analysis(self, transaction_details: Dict[str, Any], deviation: float, expected_value: float) -> Dict[str, Any]:
        """Get AI insights on valuation fraud patterns"""
        prompt = f"""
        Analyze this property valuation pattern for fraud indicators:
        
        Transaction Details: {json.dumps(transaction_details, indent=2)}
        Valuation Deviation: {deviation:.2%}
        Expected Value: ₹{expected_value:,.2f}
        
        Assess:
        1. Tax evasion scheme indicators
        2. Market rate manipulation signs
        3. Collusive undervaluation patterns
        4. Regulatory compliance risks
        
        Return JSON with keys: tax_evasion_risk, market_manipulation, collusive_patterns, compliance_risks

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in real estate valuation fraud and tax evasion detection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

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

    async def comprehensive_fraud_analysis(self, pdf_path: str) -> Dict[str, Any]:
        """Perform comprehensive fraud analysis using all detection functions"""


        document_data = await self.extract_document_info(pdf_path)
        
        analysis_results = {}
        overall_risk_score = 0.0
        fraud_indicators = []
        
        # Extract relevant data sections
        transaction_details = document_data.get("transaction_details", {})
        property_details = document_data.get("property_details", {})
        parties = document_data.get("parties", {})
        ownership_history = document_data.get("ownership_history", {})
        
        # 1. Price Deviation Analysis
        try:
            price_analysis = await self.analyze_price_deviation({
                'transaction_amount': transaction_details.get('transaction_amount'),
                'total_calculated_cost': transaction_details.get('total_calculated_cost')
            })
            analysis_results["price_deviation"] = price_analysis
            overall_risk_score = max(overall_risk_score, price_analysis.get('risk_score', 0))
            if price_analysis.get('suspicious'):
                fraud_indicators.append(f"Price deviation: {price_analysis.get('deviation_percentage', 0):.1f}%")
        except Exception as e:
            analysis_results["price_deviation"] = {"error": str(e)}
        
        # 2. Payment Concentration Analysis
        try:
            payment_analysis = await self.analyze_payment_concentration(
                transaction_details.get('payment_method', [])
            )
            analysis_results["payment_concentration"] = payment_analysis
            overall_risk_score = max(overall_risk_score, payment_analysis.get('risk_score', 0))
            if payment_analysis.get('suspicious'):
                fraud_indicators.append("Suspicious payment concentration patterns detected")
        except Exception as e:
            analysis_results["payment_concentration"] = {"error": str(e)}
        
        # 3. Rapid Ownership Transfer Analysis
        try:
            # Create ownership history from previous transaction
            ownership_list = []
            if ownership_history.get('previous_transaction'):
                ownership_list.append(ownership_history['previous_transaction'])
            
            flip_analysis = await self.detect_rapid_ownership_transfers(
                property_details.get('property_id', 'unknown'),
                ownership_list
            )
            analysis_results["rapid_transfers"] = flip_analysis
            overall_risk_score = max(overall_risk_score, flip_analysis.get('risk_score', 0))
            if flip_analysis.get('suspicious'):
                fraud_indicators.append(f"Rapid ownership transfers detected: {flip_analysis.get('rapid_transfers_count', 0)} periods")
        except Exception as e:
            analysis_results["rapid_transfers"] = {"error": str(e)}
        
        # 4. Party Identity Analysis
        try:
            identity_analysis = await self.verify_party_identity_consistency(parties)
            analysis_results["identity_consistency"] = identity_analysis
            overall_risk_score = max(overall_risk_score, identity_analysis.get('risk_score', 0))
            if identity_analysis.get('suspicious'):
                fraud_indicators.append(f"Identity inconsistencies: {len(identity_analysis.get('issues', []))} issues found")
        except Exception as e:
            analysis_results["identity_consistency"] = {"error": str(e)}
        
        # 5. Undervaluation Analysis
        try:
            underval_data = {
                'transaction_amount': transaction_details.get('transaction_amount'),
                'circle_rate_per_sqm': transaction_details.get('circle_rate_per_sqm'),
                'area_sq_meters': property_details.get('area_sq_meters'),
                'construction_rate_per_sqm': transaction_details.get('construction_rate_per_sqm')
            }
            underval_analysis = await self.analyze_undervaluation_fraud(underval_data)
            analysis_results["undervaluation"] = underval_analysis
            overall_risk_score = max(overall_risk_score, underval_analysis.get('risk_score', 0))
            if underval_analysis.get('suspicious'):
                fraud_indicators.append(f"Undervaluation detected: {underval_analysis.get('deviation_percentage', 0):.1f}% below market")
        except Exception as e:
            analysis_results["undervaluation"] = {"error": str(e)}
        
        # Determine overall fraud assessment
        if overall_risk_score >= self.risk_thresholds["high"]:
            fraud_status = "HIGH_RISK_FRAUD_SUSPECTED"
            recommendation = "IMMEDIATE_INVESTIGATION_REQUIRED"
        elif overall_risk_score >= self.risk_thresholds["medium"]:
            fraud_status = "MEDIUM_RISK_SUSPICIOUS_ACTIVITY"
            recommendation = "ENHANCED_DUE_DILIGENCE_REQUIRED"
        elif overall_risk_score >= self.risk_thresholds["low"]:
            fraud_status = "LOW_RISK_MINOR_CONCERNS"
            recommendation = "STANDARD_MONITORING_SUFFICIENT"
        else:
            fraud_status = "LOW_RISK_NO_SIGNIFICANT_CONCERNS"
            recommendation = "STANDARD_PROCESSING_ACCEPTABLE"
        
        return {
            "overall_risk_score": overall_risk_score,
            "fraud_status": fraud_status,
            "recommendation": recommendation,
            "fraud_indicators": fraud_indicators,
            "analysis_results": analysis_results,
            "timestamp": datetime.now().isoformat(),
            "property_id": property_details.get('property_id', 'unknown'),
            "transaction_amount": transaction_details.get('transaction_amount', 0)
        }

# FastAPI Service Implementation
class FraudDetectionService:
    def __init__(self):
        self.app = FastAPI(
            title="Fraud Detection Agent API",
            description="API for comprehensive real estate fraud detection and transaction pattern analysis",
            version="1.0.0"
        )
        
        # Initialize the core agent
        self.agent = PropertyFraudDetectionAgent(
            api_key=os.environ.get("PERPLEXITY_API_KEY")
        )
        
        # Agent endpoints for cross-communication
        self.agent_endpoints = {
            "document_verification": "http://localhost:8002",
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
                "agent": "fraud_detection",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/analyze-price-deviation", response_model=FraudAnalysisResult)
        async def analyze_price_deviation_endpoint(transaction_record: TransactionRecord):
            """Analyze price deviation patterns"""
            try:
                result = await self.agent.analyze_price_deviation(transaction_record.dict())
                return FraudAnalysisResult(
                    suspicious=result["suspicious"],
                    risk_score=result["risk_score"],
                    details=result,
                    analysis_type="price_deviation",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Price deviation analysis failed: {str(e)}")

        @self.app.post("/analyze-payment-concentration", response_model=FraudAnalysisResult)
        async def analyze_payment_concentration_endpoint(payment_methods: List[PaymentMethod]):
            """Analyze payment concentration patterns"""
            try:
                payment_data = [pm.dict() for pm in payment_methods]
                result = await self.agent.analyze_payment_concentration(payment_data)
                return FraudAnalysisResult(
                    suspicious=result["suspicious"],
                    risk_score=result["risk_score"],
                    details=result,
                    analysis_type="payment_concentration",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Payment concentration analysis failed: {str(e)}")

        @self.app.post("/detect-rapid-ownership-transfers", response_model=FraudAnalysisResult)
        async def detect_rapid_ownership_transfers_endpoint(property_id: str, ownership_history: List[OwnershipTransfer]):
            """Detect rapid ownership transfer patterns"""
            try:
                ownership_data = [oh.dict() for oh in ownership_history]
                result = await self.agent.detect_rapid_ownership_transfers(property_id, ownership_data)
                return FraudAnalysisResult(
                    suspicious=result["suspicious"],
                    risk_score=result["risk_score"],
                    details=result,
                    analysis_type="rapid_ownership_transfers",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Rapid ownership transfer detection failed: {str(e)}")

        @self.app.post("/verify-party-identity-consistency", response_model=FraudAnalysisResult)
        async def verify_party_identity_consistency_endpoint(parties: Parties):
            """Verify party identity consistency"""
            try:
                result = await self.agent.verify_party_identity_consistency(parties.dict())
                return FraudAnalysisResult(
                    suspicious=result["suspicious"],
                    risk_score=result["risk_score"],
                    details=result,
                    analysis_type="party_identity_consistency",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Party identity verification failed: {str(e)}")

        @self.app.post("/analyze-undervaluation-fraud", response_model=FraudAnalysisResult)
        async def analyze_undervaluation_fraud_endpoint(transaction_details: TransactionDetails):
            """Analyze undervaluation fraud patterns"""
            try:
                result = await self.agent.analyze_undervaluation_fraud(transaction_details.dict())
                return FraudAnalysisResult(
                    suspicious=result["suspicious"],
                    risk_score=result["risk_score"],
                    details=result,
                    analysis_type="undervaluation_fraud",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Undervaluation fraud analysis failed: {str(e)}")

        @self.app.post("/comprehensive-fraud-analysis")
        async def comprehensive_fraud_analysis_endpoint(pdf_path: Optional[str] = Form(None)):
            """Perform comprehensive fraud analysis"""
            try:
                result = await self.agent.comprehensive_fraud_analysis(pdf_path)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Comprehensive fraud analysis failed: {str(e)}")
        
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
        if function_name == "analyze_price_deviation":
            return await self.agent.analyze_price_deviation(payload["transaction_record"])
        elif function_name == "comprehensive_fraud_analysis":
            return await self.agent.comprehensive_fraud_analysis(payload["document_data"])
        elif function_name == "get_fraud_risk_score":
            return await self.get_fraud_risk_score(payload["property_id"])
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def get_fraud_risk_score(self, property_id: str) -> Dict[str, Any]:
        """Get fraud risk score for a property"""
        # In a real implementation, this would query cached analysis results
        return {
            "property_id": property_id,
            "risk_score": 0.3,
            "risk_level": "LOW",
            "last_updated": datetime.now().isoformat()
        }
    
    async def call_agent(self, target_agent: str, function_name: str, **kwargs) -> Any:
        """Call another agent's function"""
        if target_agent not in self.agent_endpoints:
            raise ValueError(f"Unknown agent: {target_agent}")
            
        url = f"{self.agent_endpoints[target_agent]}/cross-agent-call"
        request_data = CrossAgentRequest(
            function_name=function_name,
            payload=kwargs,
            agent_id="fraud_detection"
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
    
    def run(self, host="0.0.0.0", port=8001):
        """Run the FastAPI service"""
        uvicorn.run(self.app, host=host, port=port)

# Startup script
async def startup_fraud_service():
    """Initialize and start the fraud detection service"""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")
    
    service = FraudDetectionService()
    return service

# For testing purposes
if __name__ == "__main__":
    service = FraudDetectionService()
    service.run()
