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
class PropertyDetails(BaseModel):
    property_id: str
    property_address: str
    locality: str
    area_sq_meters: float
    floor: str
    property_type: str
    construction_year: str
    village: str
    total_land_area: str
    built_land_area: str

class TransactionDetails(BaseModel):
    transaction_amount: float
    circle_rate_per_sqm: float
    construction_rate_per_sqm: float
    calculated_land_cost: float
    calculated_construction_cost: float
    total_calculated_cost: float

class ValuationRequest(BaseModel):
    property_details: PropertyDetails
    transaction_details: TransactionDetails
    exclude_fraudulent: bool = True

class ValuationResult(BaseModel):
    fair_market_value: float
    confidence_score: float
    valuation_method: str
    details: Dict[str, Any]
    timestamp: str

class ComparableProperty(BaseModel):
    address: str
    sale_price: float
    sale_date: str
    area_sq_meters: float
    property_type: str
    price_per_sqm: float
    similarity_score: float

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

class PropertyValuationAgent:
    def __init__(self, api_key, max_concurrent_requests=5):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        self.model = "sonar"
        self.semaphore = Semaphore(max_concurrent_requests)
        
        # Valuation parameters
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Standard construction rates by type and year
        self.construction_rates = {
            "residential": {
                "2020-2025": 8000,
                "2015-2019": 7000,
                "2010-2014": 6000,
                "2005-2009": 5000,
                "before_2005": 4000
            },
            "commercial": {
                "2020-2025": 12000,
                "2015-2019": 10000,
                "2010-2014": 8500,
                "2005-2009": 7000,
                "before_2005": 6000
            }
        }
        
        # Location premium/discount factors
        self.location_factors = {
            "prime": 1.3,
            "good": 1.1,
            "average": 1.0,
            "below_average": 0.85,
            "poor": 0.7
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

    async def calculate_market_based_valuation(self, property_details: Dict[str, Any], transaction_details: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fair market value using multiple approaches"""
        
        area_sq_meters = property_details.get('area_sq_meters', 0)
        circle_rate_per_sqm = transaction_details.get('circle_rate_per_sqm', 0)
        construction_rate_per_sqm = transaction_details.get('construction_rate_per_sqm', 0)
        property_type = property_details.get('property_type', 'residential')
        locality = property_details.get('locality', '')
        
        if area_sq_meters <= 0 or circle_rate_per_sqm <= 0:
            return {
                'fair_market_value': 0,
                'confidence': 0.0,
                'error': 'Invalid property area or circle rate data',
                'method': 'calculation_failed'
            }
        
        valuations = {}
        
        # 1. Circle Rate Based Valuation (Government Rate)
        land_value = area_sq_meters * circle_rate_per_sqm
        construction_value = area_sq_meters * construction_rate_per_sqm if construction_rate_per_sqm > 0 else 0
        circle_rate_valuation = land_value + construction_value
        valuations['circle_rate_method'] = {
            'land_value': land_value,
            'construction_value': construction_value,
            'total_value': circle_rate_valuation,
            'confidence': 0.8
        }
        
        # 3. Market Approach (Comparable Sales)
        try:
            market_valuation = await self._calculate_market_approach_value(property_details)
            valuations['market_approach'] = market_valuation
        except Exception as e:
            valuations['market_approach'] = {'error': str(e), 'confidence': 0.0}
        
        # 4. Income Approach (if applicable)
        if property_type.lower() in ['commercial', 'rental', 'investment']:
            try:
                income_valuation = await self._calculate_income_approach_value(property_details, locality)
                valuations['income_approach'] = income_valuation
            except Exception as e:
                valuations['income_approach'] = {'error': str(e), 'confidence': 0.0}
        
        # Calculate weighted average based on confidence scores
        weighted_value = 0
        total_weight = 0
        
        for method, valuation in valuations.items():
            if 'total_value' in valuation and 'confidence' in valuation:
                confidence = valuation['confidence']
                value = valuation['total_value']
                weighted_value += value * confidence
                total_weight += confidence
        
        final_market_value = weighted_value / total_weight if total_weight > 0 else circle_rate_valuation
        overall_confidence = total_weight / len([v for v in valuations.values() if 'confidence' in v])
        
        # Get AI enhancement for complex cases
        ai_insights = {}
        if overall_confidence < 0.7 or any('error' in v for v in valuations.values()):
            ai_insights = await self._get_ai_valuation_insights(property_details, transaction_details, valuations)
        
        return {
            'fair_market_value': final_market_value,
            'confidence': min(1.0, overall_confidence),
            'valuations_breakdown': valuations,
            'methodology': 'weighted_multi_approach',
            'ai_insights': ai_insights,
            'price_per_sqm': final_market_value / area_sq_meters if area_sq_meters > 0 else 0
        }

    async def analyze_comparable_properties(self, property_details: Dict[str, Any], exclude_fraudulent: bool = True) -> Dict[str, Any]:
        """Find and analyze similar property transactions"""
        
        locality = property_details.get('locality', '')
        area_sq_meters = property_details.get('area_sq_meters', 0)
        property_type = property_details.get('property_type', '')
        property_address = property_details.get('property_address', '')
        
        # Get comparable properties using AI search
        comparables = await self._find_comparable_properties_ai(property_details)
        
        # Filter fraudulent transactions if requested
        if exclude_fraudulent:
            try:
                fraudulent_list = await self._get_fraudulent_transactions_list(locality)
                comparables = [comp for comp in comparables if comp.get('transaction_id') not in fraudulent_list]
            except Exception as e:
                print(f"Could not filter fraudulent transactions: {e}")
        
        if not comparables:
            return {
                'comparables': [],
                'average_price_per_sqm': 0,
                'confidence': 0.0,
                'error': 'No comparable properties found'
            }
        
        # Calculate similarity scores and adjust prices
        processed_comparables = []
        total_adjusted_value = 0
        total_weight = 0
        
        for comp in comparables:
            try:
                similarity_score = self._calculate_similarity_score(property_details, comp)
                adjusted_price = self._adjust_comparable_price(property_details, comp)
                
                processed_comp = {
                    'address': comp.get('address', 'Unknown'),
                    'sale_price': comp.get('sale_price', 0),
                    'sale_date': comp.get('sale_date', ''),
                    'area_sq_meters': comp.get('area_sq_meters', 0),
                    'property_type': comp.get('property_type', ''),
                    'price_per_sqm': comp.get('sale_price', 0) / comp.get('area_sq_meters', 1),
                    'adjusted_price': adjusted_price,
                    'similarity_score': similarity_score,
                    'adjustments_made': comp.get('adjustments', [])
                }
                
                processed_comparables.append(processed_comp)
                total_adjusted_value += adjusted_price * similarity_score
                total_weight += similarity_score
                
            except Exception as e:
                continue
        
        # Calculate market value based on comparables
        market_value_from_comparables = total_adjusted_value / total_weight if total_weight > 0 else 0
        average_price_per_sqm = market_value_from_comparables / area_sq_meters if area_sq_meters > 0 else 0
        
        # Confidence based on number and quality of comparables
        confidence = min(1.0, (len(processed_comparables) * 0.2) + (total_weight / len(processed_comparables) * 0.6) if processed_comparables else 0)
        
        return {
            'comparables': processed_comparables,
            'market_value_estimate': market_value_from_comparables,
            'average_price_per_sqm': average_price_per_sqm,
            'confidence': confidence,
            'comparable_count': len(processed_comparables),
            'analysis_date': datetime.now().isoformat()
        }

    async def assess_construction_cost_accuracy(self, property_details: Dict[str, Any], calculated_construction_cost: float) -> Dict[str, Any]:
        """Validate construction cost calculations"""
        
        area_sq_meters = property_details.get('area_sq_meters', 0)
        construction_year = property_details.get('construction_year', '')
        property_type = property_details.get('property_type', 'residential')
        floor = property_details.get('floor', '')
        
        if area_sq_meters <= 0:
            return {
                'accurate': False,
                'confidence': 0.0,
                'error': 'Invalid property area for construction cost assessment'
            }
        
        # Determine standard construction rate
        property_type_clean = property_type.lower()
        if 'commercial' in property_type_clean:
            rate_category = 'commercial'
        else:
            rate_category = 'residential'
        
        # Determine year category
        try:
            if 'after' in construction_year.lower():
                year_num = int(re.findall(r'\d{4}', construction_year)[0])
            else:
                year_num = int(construction_year) if construction_year.isdigit() else 2010
        except (IndexError, ValueError):
            year_num = 2010
        
        if year_num >= 2020:
            year_category = "2020-2025"
        elif year_num >= 2015:
            year_category = "2015-2019"
        elif year_num >= 2010:
            year_category = "2010-2014"
        elif year_num >= 2005:
            year_category = "2005-2009"
        else:
            year_category = "before_2005"
        
        standard_rate = self.construction_rates.get(rate_category, {}).get(year_category, 6000)
        
        # Apply floor adjustments
        floor_multiplier = 1.0
        if 'ground' in floor.lower():
            floor_multiplier = 0.95
        elif 'first' in floor.lower():
            floor_multiplier = 1.0
        elif 'second' in floor.lower() or 'third' in floor.lower():
            floor_multiplier = 1.05
        elif any(word in floor.lower() for word in ['fourth', 'fifth', 'higher']):
            floor_multiplier = 1.1
        
        adjusted_standard_rate = standard_rate * floor_multiplier
        expected_construction_cost = area_sq_meters * adjusted_standard_rate
        
        # Calculate deviation
        if expected_construction_cost > 0:
            deviation = abs(calculated_construction_cost - expected_construction_cost) / expected_construction_cost
        else:
            deviation = 1.0
        
        # Assess accuracy
        if deviation <= 0.15:  # Within 15%
            accurate = True
            confidence = 0.9
            assessment = "ACCURATE"
        elif deviation <= 0.25:  # Within 25%
            accurate = True
            confidence = 0.7
            assessment = "REASONABLY_ACCURATE"
        elif deviation <= 0.40:  # Within 40%
            accurate = False
            confidence = 0.5
            assessment = "QUESTIONABLE"
        else:
            accurate = False
            confidence = 0.2
            assessment = "INACCURATE"
        
        # Get AI insights for significant deviations
        ai_insights = {}
        if deviation > 0.25:
            ai_insights = await self._get_ai_construction_analysis(property_details, calculated_construction_cost, expected_construction_cost)
        
        return {
            'accurate': accurate,
            'confidence': confidence,
            'assessment': assessment,
            'calculated_cost': calculated_construction_cost,
            'expected_cost': expected_construction_cost,
            'deviation_percentage': deviation * 100,
            'standard_rate_used': adjusted_standard_rate,
            'rate_category': rate_category,
            'year_category': year_category,
            'floor_adjustment': floor_multiplier,
            'ai_insights': ai_insights
        }

    async def calculate_location_premium_discount(self, locality: str, village: str, property_address: str) -> Dict[str, Any]:
        """Assess location-based value adjustments"""
        
        # Get AI-powered location analysis
        location_analysis = await self._get_ai_location_analysis(locality, village, property_address)
        
        # Analyze location factors
        factors = {
            'connectivity': 0.0,
            'amenities': 0.0,
            'infrastructure': 0.0,
            'safety': 0.0,
            'development_potential': 0.0,
            'market_demand': 0.0
        }
        
        # Extract factor scores from AI analysis
        if 'location_factors' in location_analysis:
            ai_factors = location_analysis['location_factors']
            for factor in factors:
                if factor in ai_factors:
                    try:
                        factors[factor] = float(ai_factors[factor])
                    except (ValueError, TypeError):
                        factors[factor] = 0.5  # Default neutral score
        
        # Calculate weighted location score
        weights = {
            'connectivity': 0.25,
            'amenities': 0.20,
            'infrastructure': 0.20,
            'safety': 0.15,
            'development_potential': 0.10,
            'market_demand': 0.10
        }
        
        location_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        # Determine premium/discount
        if location_score >= 0.8:
            multiplier = self.location_factors['prime']
            category = "PRIME_LOCATION"
        elif location_score >= 0.65:
            multiplier = self.location_factors['good']
            category = "GOOD_LOCATION"
        elif location_score >= 0.45:
            multiplier = self.location_factors['average']
            category = "AVERAGE_LOCATION"
        elif location_score >= 0.3:
            multiplier = self.location_factors['below_average']
            category = "BELOW_AVERAGE_LOCATION"
        else:
            multiplier = self.location_factors['poor']
            category = "POOR_LOCATION"
        
        premium_discount_percentage = (multiplier - 1.0) * 100
        
        return {
            'location_multiplier': multiplier,
            'premium_discount_percentage': premium_discount_percentage,
            'location_category': category,
            'location_score': location_score,
            'factor_breakdown': factors,
            'ai_analysis': location_analysis,
            'confidence': location_analysis.get('confidence', 0.7)
        }

    async def detect_value_manipulation_indicators(self, transaction_record: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential value manipulation"""
        
        transaction_amount = transaction_record.get('transaction_amount', 0)
        property_details = transaction_record.get('property_details', {})
        transaction_details = transaction_record.get('transaction_details', {})
        
        # Calculate fair market value
        market_valuation = await self.calculate_market_based_valuation(property_details, transaction_details)
        fair_market_value = market_valuation.get('fair_market_value', 0)
        
        if fair_market_value <= 0:
            return {
                'manipulation_detected': False,
                'confidence': 0.0,
                'error': 'Could not determine fair market value for comparison'
            }
        
        # Calculate deviation
        deviation = (transaction_amount - fair_market_value) / fair_market_value
        manipulation_indicators = []
        risk_score = 0.0
        
        # Check for undervaluation (common for tax evasion)
        if deviation < -0.3:  # 30% below market value
            manipulation_indicators.append({
                'type': 'significant_undervaluation',
                'severity': 'HIGH',
                'deviation_percentage': deviation * 100,
                'potential_motive': 'tax_evasion'
            })
            risk_score = max(risk_score, 0.8)
        elif deviation < -0.15:  # 15% below market value
            manipulation_indicators.append({
                'type': 'moderate_undervaluation',
                'severity': 'MEDIUM',
                'deviation_percentage': deviation * 100,
                'potential_motive': 'tax_optimization'
            })
            risk_score = max(risk_score, 0.6)
        
        # Check for overvaluation (common for money laundering)
        if deviation > 0.5:  # 50% above market value
            manipulation_indicators.append({
                'type': 'significant_overvaluation',
                'severity': 'HIGH',
                'deviation_percentage': deviation * 100,
                'potential_motive': 'money_laundering'
            })
            risk_score = max(risk_score, 0.9)
        elif deviation > 0.25:  # 25% above market value
            manipulation_indicators.append({
                'type': 'moderate_overvaluation',
                'severity': 'MEDIUM',
                'deviation_percentage': deviation * 100,
                'potential_motive': 'value_inflation'
            })
            risk_score = max(risk_score, 0.7)
        
        # Check valuation confidence
        valuation_confidence = market_valuation.get('confidence', 0)
        if valuation_confidence < 0.5:
            manipulation_indicators.append({
                'type': 'unreliable_market_valuation',
                'severity': 'MEDIUM',
                'confidence': valuation_confidence,
                'potential_issue': 'insufficient_comparable_data'
            })
            risk_score = max(risk_score, 0.5)
        
        # Get AI insights for complex manipulation patterns
        ai_insights = {}
        if risk_score >= 0.6:
            ai_insights = await self._get_ai_manipulation_analysis(transaction_record, fair_market_value, deviation)
        
        manipulation_detected = len(manipulation_indicators) > 0 and risk_score >= 0.5
        
        return {
            'manipulation_detected': manipulation_detected,
            'risk_score': risk_score,
            'transaction_amount': transaction_amount,
            'fair_market_value': fair_market_value,
            'deviation_amount': transaction_amount - fair_market_value,
            'deviation_percentage': deviation * 100,
            'indicators': manipulation_indicators,
            'valuation_confidence': valuation_confidence,
            'ai_insights': ai_insights
        }

    async def validate_circle_rate_compliance(self, transaction_details: Dict[str, Any], property_details: Dict[str, Any]) -> Dict[str, Any]:
        """Verify transaction aligns with government circle rates"""
        
        transaction_amount = transaction_details.get('transaction_amount', 0)
        circle_rate_per_sqm = transaction_details.get('circle_rate_per_sqm', 0)
        area_sq_meters = property_details.get('area_sq_meters', 0)
        locality = property_details.get('locality', '')
        
        if circle_rate_per_sqm <= 0 or area_sq_meters <= 0:
            return {
                'compliant': False,
                'confidence': 0.0,
                'error': 'Invalid circle rate or property area data'
            }
        
        # Calculate minimum transaction value based on circle rate
        minimum_land_value = area_sq_meters * circle_rate_per_sqm
        
        # Add construction component if available
        construction_rate_per_sqm = transaction_details.get('construction_rate_per_sqm', 0)
        minimum_construction_value = area_sq_meters * construction_rate_per_sqm if construction_rate_per_sqm > 0 else 0
        
        minimum_total_value = minimum_land_value + minimum_construction_value
        
        # Check compliance
        if transaction_amount >= minimum_total_value:
            compliant = True
            compliance_status = "COMPLIANT"
            risk_level = "LOW"
        else:
            compliant = False
            shortfall = minimum_total_value - transaction_amount
            shortfall_percentage = (shortfall / minimum_total_value) * 100
            
            if shortfall_percentage > 25:
                compliance_status = "SEVERELY_NON_COMPLIANT"
                risk_level = "HIGH"
            elif shortfall_percentage > 10:
                compliance_status = "MODERATELY_NON_COMPLIANT"
                risk_level = "MEDIUM"
            else:
                compliance_status = "SLIGHTLY_NON_COMPLIANT"
                risk_level = "LOW"
        
        # Get current circle rates for validation
        circle_rate_validation = await self._validate_circle_rate_accuracy(locality, circle_rate_per_sqm)
        
        # Calculate tax implications
        stamp_duty_on_transaction = transaction_amount * 0.05  # Assume 5% stamp duty
        stamp_duty_on_circle_rate = minimum_total_value * 0.05
        tax_difference = stamp_duty_on_circle_rate - stamp_duty_on_transaction
        
        return {
            'compliant': compliant,
            'compliance_status': compliance_status,
            'risk_level': risk_level,
            'transaction_amount': transaction_amount,
            'minimum_circle_rate_value': minimum_total_value,
            'shortfall_amount': max(0, minimum_total_value - transaction_amount),
            'shortfall_percentage': max(0, ((minimum_total_value - transaction_amount) / minimum_total_value) * 100),
            'breakdown': {
                'minimum_land_value': minimum_land_value,
                'minimum_construction_value': minimum_construction_value,
                'circle_rate_per_sqm': circle_rate_per_sqm,
                'construction_rate_per_sqm': construction_rate_per_sqm
            },
            'tax_implications': {
                'stamp_duty_on_transaction': stamp_duty_on_transaction,
                'stamp_duty_on_circle_rate': stamp_duty_on_circle_rate,
                'potential_tax_difference': tax_difference
            },
            'circle_rate_validation': circle_rate_validation,
            'confidence': 0.9 if circle_rate_validation.get('accurate', True) else 0.6
        }

    # AI Enhancement Methods
    async def _get_ai_valuation_insights(self, property_details: Dict[str, Any], transaction_details: Dict[str, Any], valuations: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI insights on complex valuation scenarios"""
        prompt = f"""
        Analyze this property valuation scenario and provide expert insights:
        
        Property Details: {json.dumps(property_details, indent=2)}
        Transaction Details: {json.dumps(transaction_details, indent=2)}
        Valuation Breakdown: {json.dumps(valuations, indent=2)}
        
        Provide insights on:
        1. Which valuation method is most reliable for this property type and location
        2. Factors that might explain valuation discrepancies
        3. Market conditions affecting this property's value
        4. Recommended adjustments to improve valuation accuracy
        
        Return JSON with keys: recommended_method, discrepancy_factors, market_conditions, adjustments

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert real estate appraiser with deep knowledge of Indian property markets and valuation methodologies."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    #
    async def _find_comparable_properties_ai(self, property_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to find comparable properties"""
        prompt = f"""
        Find comparable property transactions for this property:
        
        {json.dumps(property_details, indent=2)}
        
        Search for properties with similar:
        - Location (same locality or nearby areas)
        - Size (±20% area)
        - Type and construction year
        - Transaction date (within last 2 years)
        
        Return JSON array with fields: address, sale_price, sale_date, area_sq_meters, property_type, transaction_id

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a real estate data analyst who finds comparable property transactions from public records and databases."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            result = self._parse_json_response(response.choices[0].message.content)
            return result if isinstance(result, list) else result.get('comparables', [])
        except Exception as e:
            # Return mock comparable data as fallback
            return self._generate_mock_comparables(property_details)

    def _generate_mock_comparables(self, property_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mock comparable properties for testing"""
        base_price = property_details.get('area_sq_meters', 50) * 25000  # Base rate ₹25,000 per sqm
        comparables = []
        
        for i in range(5):
            area_variation = np.random.uniform(0.8, 1.2)
            price_variation = np.random.uniform(0.9, 1.1)
            
            comparable = {
                'address': f"Property {i+1}, {property_details.get('locality', 'Area')}",
                'sale_price': base_price * price_variation,
                'sale_date': (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d'),
                'area_sq_meters': property_details.get('area_sq_meters', 50) * area_variation,
                'property_type': property_details.get('property_type', 'residential'),
                'transaction_id': f"MOCK_TX_{i+1}"
            }
            comparables.append(comparable)
        
        return comparables

    def _calculate_similarity_score(self, subject_property: Dict[str, Any], comparable: Dict[str, Any]) -> float:
        """Calculate similarity score between subject property and comparable"""
        score = 0.0
        
        # Area similarity (40% weight)
        subject_area = subject_property.get('area_sq_meters', 0)
        comp_area = comparable.get('area_sq_meters', 0)
        if subject_area > 0 and comp_area > 0:
            area_ratio = min(subject_area, comp_area) / max(subject_area, comp_area)
            score += area_ratio * 0.4
        
        # Property type similarity (30% weight)
        if subject_property.get('property_type', '').lower() == comparable.get('property_type', '').lower():
            score += 0.3
        
        # Location similarity (20% weight) - simplified check
        subject_locality = subject_property.get('locality', '').lower()
        comp_address = comparable.get('address', '').lower()
        if subject_locality in comp_address:
            score += 0.2
        elif any(word in comp_address for word in subject_locality.split()):
            score += 0.1
        
        # Date recency (10% weight)
        try:
            sale_date = datetime.strptime(comparable.get('sale_date', '2020-01-01'), '%Y-%m-%d')
            days_old = (datetime.now() - sale_date).days
            if days_old <= 180:  # Within 6 months
                score += 0.1
            elif days_old <= 365:  # Within 1 year
                score += 0.05
        except:
            pass
        
        return min(1.0, score)

    def _adjust_comparable_price(self, subject_property: Dict[str, Any], comparable: Dict[str, Any]) -> float:
        """Adjust comparable price for differences with subject property"""
        base_price = comparable.get('sale_price', 0)
        
        # Area adjustment
        subject_area = subject_property.get('area_sq_meters', 0)
        comp_area = comparable.get('area_sq_meters', 0)
        if comp_area > 0:
            price_per_sqm = base_price / comp_area
            adjusted_price = price_per_sqm * subject_area
        else:
            adjusted_price = base_price
        
        # Time adjustment (assume 5% annual appreciation)
        try:
            sale_date = datetime.strptime(comparable.get('sale_date', '2023-01-01'), '%Y-%m-%d')
            years_old = (datetime.now() - sale_date).days / 365.25
            time_adjustment = (1.05 ** years_old)
            adjusted_price *= time_adjustment
        except:
            pass
        
        return adjusted_price

    async def _calculate_market_approach_value(self, property_details: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate value using market approach method"""
        # Get comparable sales and calculate average
        comparables = await self._find_comparable_properties_ai(property_details)
        
        if not comparables:
            return {'error': 'No comparable sales found', 'confidence': 0.0}
        
        total_price_per_sqm = 0
        count = 0
        
        for comp in comparables:
            if comp.get('area_sq_meters', 0) > 0:
                price_per_sqm = comp.get('sale_price', 0) / comp.get('area_sq_meters', 1)
                total_price_per_sqm += price_per_sqm
                count += 1
        
        if count > 0:
            avg_price_per_sqm = total_price_per_sqm / count
            total_value = avg_price_per_sqm * property_details.get('area_sq_meters', 0)
            confidence = min(1.0, count * 0.2)  # Higher confidence with more comparables
        else:
            total_value = 0
            confidence = 0.0
        
        return {
            'total_value': total_value,
            'avg_price_per_sqm': avg_price_per_sqm if count > 0 else 0,
            'comparable_count': count,
            'confidence': confidence,
            'method': 'market_approach'
        }

    async def _calculate_income_approach_value(self, property_details: Dict[str, Any], locality: str) -> Dict[str, Any]:
        """Calculate value using income approach method"""
        # Simplified income approach calculation
        area = property_details.get('area_sq_meters', 0)
        
        # Estimate rental yield based on property type and location
        if 'commercial' in property_details.get('property_type', '').lower():
            monthly_rent_per_sqm = 150  # ₹150 per sqm for commercial
            cap_rate = 0.08  # 8% capitalization rate
        else:
            monthly_rent_per_sqm = 80   # ₹80 per sqm for residential
            cap_rate = 0.06  # 6% capitalization rate
        
        annual_rent = area * monthly_rent_per_sqm * 12
        total_value = annual_rent / cap_rate
        
        return {
            'total_value': total_value,
            'annual_rent': annual_rent,
            'cap_rate': cap_rate,
            'confidence': 0.6,
            'method': 'income_approach'
        }

    async def _get_ai_location_analysis(self, locality: str, village: str, property_address: str) -> Dict[str, Any]:
        """Get AI analysis of location factors"""
        prompt = f"""
        Analyze the location factors for this property address:
        
        Locality: {locality}
        Village: {village}
        Full Address: {property_address}
        
        Rate each factor from 0.0 to 1.0:
        1. Connectivity (public transport, roads, highways)
        2. Amenities (schools, hospitals, shopping)
        3. Infrastructure (water, electricity, sewage)
        4. Safety (crime rates, neighborhood safety)
        5. Development potential (future growth prospects)
        6. Market demand (buyer interest, liquidity)
        
        Return JSON with location_factors object and overall assessment

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a location analysis expert familiar with Indian urban and rural property markets."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            # Return default neutral scores
            return {
                'location_factors': {
                    'connectivity': 0.5,
                    'amenities': 0.5,
                    'infrastructure': 0.5,
                    'safety': 0.5,
                    'development_potential': 0.5,
                    'market_demand': 0.5
                },
                'confidence': 0.5,
                'error': str(e)
            }

    async def _get_ai_construction_analysis(self, property_details: Dict[str, Any], calculated_cost: float, expected_cost: float) -> Dict[str, Any]:
        """Get AI analysis of construction cost discrepancies"""
        prompt = f"""
        Analyze this construction cost discrepancy:
        
        Property: {json.dumps(property_details, indent=2)}
        Calculated Cost: ₹{calculated_cost:,.2f}
        Expected Cost: ₹{expected_cost:,.2f}
        
        Explain possible reasons for the difference and assess validity.
        Return JSON with: reasons, validity_assessment, recommendations
        
        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a construction cost expert familiar with Indian building materials and labor costs."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _get_ai_manipulation_analysis(self, transaction_record: Dict[str, Any], fair_market_value: float, deviation: float) -> Dict[str, Any]:
        """Get AI analysis of potential value manipulation"""
        prompt = f"""
        Analyze this potential value manipulation case:
        
        Transaction: {json.dumps(transaction_record, indent=2)}
        Fair Market Value: ₹{fair_market_value:,.2f}
        Deviation: {deviation:.2%}
        
        Assess manipulation likelihood and provide investigation recommendations.
        Return JSON with: manipulation_likelihood, investigation_steps, red_flags

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a forensic real estate analyst expert in detecting value manipulation and fraud."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

    async def _validate_circle_rate_accuracy(self, locality: str, provided_rate: float) -> Dict[str, Any]:
        """Validate circle rate against current government rates"""
        prompt = f"""
        Verify the circle rate for this location:
        
        Locality: {locality}
        Provided Rate: ₹{provided_rate:,.2f} per sq.m
        
        Check if this rate is reasonable for the location and current year.
        Return JSON with: accurate, current_rate_range, confidence

        keep it accurate and conscise
        """
        
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert on Indian government circle rates and property valuations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            return {"accurate": True, "confidence": 0.5, "error": str(e)}

    async def _get_fraudulent_transactions_list(self, area: str) -> List[str]:
        """Get list of known fraudulent transactions in the area"""
        # This would typically call the fraud detection agent
        # For now, return an empty list
        return []

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
            

    async def comprehensive_property_valuation(self, pdf_path : str) -> Dict[str, Any]:
        """Perform comprehensive property valuation using all methods"""


        document_data = await self.extract_document_info(pdf_path)

        # Extract relevant data sections
        transaction_details = document_data.get("transaction_details", {})
        property_details = document_data.get("property_details", {})
        
        valuation_results = {}
        
        # 1. Market-based valuation
        try:
            market_valuation = await self.calculate_market_based_valuation(property_details, transaction_details)
            valuation_results["market_valuation"] = market_valuation
        except Exception as e:
            valuation_results["market_valuation"] = {"error": str(e)}
        
        # 2. Comparable properties analysis
        try:
            comparable_analysis = await self.analyze_comparable_properties(property_details, exclude_fraudulent=True)
            valuation_results["comparable_analysis"] = comparable_analysis
        except Exception as e:
            valuation_results["comparable_analysis"] = {"error": str(e)}
        
        # 3. Construction cost assessment
        try:
            construction_assessment = await self.assess_construction_cost_accuracy(
                property_details, 
                transaction_details.get('calculated_construction_cost', 0)
            )
            valuation_results["construction_assessment"] = construction_assessment
        except Exception as e:
            valuation_results["construction_assessment"] = {"error": str(e)}
        
        # 4. Location premium/discount
        try:
            location_analysis = await self.calculate_location_premium_discount(
                property_details.get('locality', ''),
                property_details.get('village', ''),
                property_details.get('property_address', '')
            )
            valuation_results["location_analysis"] = location_analysis
        except Exception as e:
            valuation_results["location_analysis"] = {"error": str(e)}
        
        # 5. Value manipulation detection
        try:
            manipulation_check = await self.detect_value_manipulation_indicators({
                'transaction_amount': transaction_details.get('transaction_amount', 0),
                'property_details': property_details,
                'transaction_details': transaction_details
            })
            valuation_results["manipulation_check"] = manipulation_check
        except Exception as e:
            valuation_results["manipulation_check"] = {"error": str(e)}
        
        # 6. Circle rate compliance
        try:
            circle_rate_compliance = await self.validate_circle_rate_compliance(transaction_details, property_details)
            valuation_results["circle_rate_compliance"] = circle_rate_compliance
        except Exception as e:
            valuation_results["circle_rate_compliance"] = {"error": str(e)}
        
        # Calculate final valuation summary
        market_value = valuation_results.get("market_valuation", {}).get("fair_market_value", 0)
        comparable_value = valuation_results.get("comparable_analysis", {}).get("market_value_estimate", 0)
        
        # Use market valuation as primary, comparable as secondary
        if market_value > 0:
            final_valuation = market_value
            confidence = valuation_results.get("market_valuation", {}).get("confidence", 0.5)
        elif comparable_value > 0:
            final_valuation = comparable_value
            confidence = valuation_results.get("comparable_analysis", {}).get("confidence", 0.5)
        else:
            final_valuation = transaction_details.get('total_calculated_cost', 0)
            confidence = 0.3
        
        return {
            "final_valuation": final_valuation,
            "confidence_score": confidence,
            "valuation_breakdown": valuation_results,
            "property_id": property_details.get('property_id', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "transaction_amount": transaction_details.get('transaction_amount', 0),
            "manipulation_detected": valuation_results.get("manipulation_check", {}).get("manipulation_detected", False)
        }

# FastAPI Service Implementation
class PropertyValuationService:
    def __init__(self):
        self.app = FastAPI(
            title="Property Valuation Agent API",
            description="API for comprehensive real estate property valuation and market analysis",
            version="1.0.0"
        )
        
        # Initialize the core agent
        self.agent = PropertyValuationAgent(
            api_key=os.environ.get("PERPLEXITY_API_KEY")
        )
        
        # Agent endpoints for cross-communication
        self.agent_endpoints = {
            "fraud_detection": "http://localhost:8001",
            "document_verification": "http://localhost:8002"
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
                "agent": "property_valuation",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/calculate-market-valuation", response_model=ValuationResult)
        async def calculate_market_valuation_endpoint(request: ValuationRequest):
            """Calculate market-based property valuation"""
            try:
                result = await self.agent.calculate_market_based_valuation(
                    request.property_details.dict(), 
                    request.transaction_details.dict()
                )
                return ValuationResult(
                    fair_market_value=result["fair_market_value"],
                    confidence_score=result["confidence"],
                    valuation_method="market_based_multi_approach",
                    details=result,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Market valuation failed: {str(e)}")

        @self.app.post("/analyze-comparable-properties")
        async def analyze_comparable_properties_endpoint(property_details: PropertyDetails, exclude_fraudulent: bool = True):
            """Analyze comparable properties"""
            try:
                result = await self.agent.analyze_comparable_properties(property_details.dict(), exclude_fraudulent)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Comparable analysis failed: {str(e)}")

        @self.app.post("/assess-construction-cost")
        async def assess_construction_cost_endpoint(property_details: PropertyDetails, calculated_construction_cost: float):
            """Assess construction cost accuracy"""
            try:
                result = await self.agent.assess_construction_cost_accuracy(property_details.dict(), calculated_construction_cost)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Construction cost assessment failed: {str(e)}")

        @self.app.post("/calculate-location-adjustment")
        async def calculate_location_adjustment_endpoint(locality: str, village: str, property_address: str):
            """Calculate location-based premium/discount"""
            try:
                result = await self.agent.calculate_location_premium_discount(locality, village, property_address)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Location analysis failed: {str(e)}")

        @self.app.post("/detect-value-manipulation")
        async def detect_value_manipulation_endpoint(transaction_record: Dict[str, Any]):
            """Detect potential value manipulation"""
            try:
                result = await self.agent.detect_value_manipulation_indicators(transaction_record)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Value manipulation detection failed: {str(e)}")

        @self.app.post("/validate-circle-rate-compliance")
        async def validate_circle_rate_compliance_endpoint(transaction_details: TransactionDetails, property_details: PropertyDetails):
            """Validate circle rate compliance"""
            try:
                result = await self.agent.validate_circle_rate_compliance(transaction_details.dict(), property_details.dict())
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Circle rate validation failed: {str(e)}")

        @self.app.post("/comprehensive-valuation")
        async def comprehensive_valuation_endpoint(pdf_path: Optional[str] = Form(None)):
            """Perform comprehensive property valuation"""
            try:
                result = await self.agent.comprehensive_property_valuation(pdf_path)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Comprehensive valuation failed: {str(e)}")
        
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
        if function_name == "calculate_market_based_valuation":
            return await self.agent.calculate_market_based_valuation(
                payload["property_details"], 
                payload["transaction_details"]
            )
        elif function_name == "detect_value_manipulation_indicators":
            return await self.agent.detect_value_manipulation_indicators(payload["transaction_record"])
        elif function_name == "comprehensive_property_valuation":
            return await self.agent.comprehensive_property_valuation(pdf_str)
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def call_agent(self, target_agent: str, function_name: str, **kwargs) -> Any:
        """Call another agent's function"""
        if target_agent not in self.agent_endpoints:
            raise ValueError(f"Unknown agent: {target_agent}")
            
        url = f"{self.agent_endpoints[target_agent]}/cross-agent-call"
        request_data = CrossAgentRequest(
            function_name=function_name,
            payload=kwargs,
            agent_id="property_valuation"
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
    
    def run(self, host="0.0.0.0", port=8003):
        """Run the FastAPI service"""
        uvicorn.run(self.app, host=host, port=port )

# Startup script
async def startup_valuation_service():
    """Initialize and start the property valuation service"""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")
    
    service = PropertyValuationService()
    return service

# For testing purposes
if __name__ == "__main__":
    service = PropertyValuationService()
    service.run()
