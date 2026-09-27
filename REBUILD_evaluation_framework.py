"""
Evaluation Framework for MedTriageAI+

Comprehensive testing and validation framework for distinction-level assessment
"""

import json
from datetime import datetime
from pathlib import Path

class EvaluationFramework:
    """
    Framework for evaluating MedTriageAI+ system performance
    Used for distinction-level user testing and iteration
    """
    
    def __init__(self):
        self.test_cases = []
        self.results = []
        self.feedback = []
    
    def add_test_case(self, patient_data, expected_ktas):
        """
        Add a test case for evaluation
        
        Args:
            patient_data: Patient vitals and symptoms
            expected_ktas: Expected KTAS level (ground truth)
        """
        self.test_cases.append({
            'timestamp': datetime.now().isoformat(),
            'patient_data': patient_data,
            'expected_ktas': expected_ktas,
            'test_id': len(self.test_cases) + 1
        })
    
    def record_prediction(self, test_id, predicted_ktas, confidence):
        """Record a system prediction"""
        self.results.append({
            'test_id': test_id,
            'predicted_ktas': predicted_ktas,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_feedback(self, test_id, user_feedback, user_type='staff'):
        """
        Record user feedback for iteration
        DISTINCTION-LEVEL: Shows iteration based on feedback
        """
        self.feedback.append({
            'test_id': test_id,
            'user_type': user_type,
            'feedback_text': user_feedback,
            'was_helpful': None,  # To be filled
            'suggestions': [],
            'timestamp': datetime.now().isoformat()
        })
    
    def calculate_accuracy(self):
        """Calculate overall accuracy"""
        if not self.results:
            return 0.0
        
        correct = 0
        for result in self.results:
            test_id = result['test_id']
            test = next((t for t in self.test_cases if t['test_id'] == test_id), None)
            if test and result['predicted_ktas'] == test['expected_ktas']:
                correct += 1
        
        return correct / len(self.results) * 100
    
    def per_class_accuracy(self):
        """Calculate accuracy per KTAS class"""
        accuracy_per_class = {}
        
        for ktas_level in range(1, 6):
            test_cases = [t for t in self.test_cases if t['expected_ktas'] == ktas_level]
            if not test_cases:
                continue
            
            correct = 0
            for test in test_cases:
                result = next((r for r in self.results if r['test_id'] == test['test_id']), None)
                if result and result['predicted_ktas'] == ktas_level:
                    correct += 1
            
            accuracy = (correct / len(test_cases) * 100) if test_cases else 0
            accuracy_per_class[f'KTAS {ktas_level}'] = accuracy
        
        return accuracy_per_class
    
    def safety_metrics(self):
        """Calculate safety-related metrics"""
        undertriage = 0  # Assigned lower priority than needed (dangerous)
        overtriage = 0   # Assigned higher priority than needed (safe but inefficient)
        correct = 0
        
        for result in self.results:
            test = next((t for t in self.test_cases if t['test_id'] == result['test_id']), None)
            if not test:
                continue
            
            expected = test['expected_ktas']
            predicted = result['predicted_ktas']
            
            if predicted == expected:
                correct += 1
            elif predicted < expected:  # DANGEROUS: patient gets lower priority
                undertriage += 1
            else:  # Safe: patient gets higher priority
                overtriage += 1
        
        total = len(self.results)
        
        return {
            'correct': correct,
            'undertriage_count': undertriage,
            'undertriage_rate': (undertriage / total * 100) if total > 0 else 0,
            'overtriage_count': overtriage,
            'overtriage_rate': (overtriage / total * 100) if total > 0 else 0,
            'safety_score': 100 - (undertriage / total * 100) if total > 0 else 0
        }
    
    def confidence_calibration(self):
        """
        Check if confidence scores match accuracy
        (Confidence calibration = system's confidence matches actual accuracy)
        """
        if not self.results:
            return 0.0
        
        correct = 0
        total_confidence = 0
        
        for result in self.results:
            test = next((t for t in self.test_cases if t['test_id'] == result['test_id']), None)
            if test and result['predicted_ktas'] == test['expected_ktas']:
                correct += 1
            total_confidence += result['confidence']
        
        actual_accuracy = (correct / len(self.results)) if self.results else 0
        avg_confidence = total_confidence / len(self.results) if self.results else 0
        
        # Calibration error: difference between confidence and accuracy
        calibration_error = abs(avg_confidence - actual_accuracy)
        
        return {
            'actual_accuracy': actual_accuracy,
            'average_confidence': avg_confidence,
            'calibration_error': calibration_error,
            'is_well_calibrated': calibration_error < 0.1  # Within 10%
        }
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_count': len(self.test_cases),
            'overall_accuracy': f"{self.calculate_accuracy():.1f}%",
            'per_class_accuracy': self.per_class_accuracy(),
            'safety_metrics': self.safety_metrics(),
            'confidence_calibration': self.confidence_calibration(),
            'feedback_summary': self._summarize_feedback()
        }
        return report
    
    def _summarize_feedback(self):
        """Summarize user feedback"""
        if not self.feedback:
            return "No feedback collected"
        
        helpful_count = sum(1 for f in self.feedback if f.get('was_helpful'))
        total_feedback = len(self.feedback)
        
        return {
            'total_feedback': total_feedback,
            'helpful_feedback': helpful_count,
            'helpfulness_rate': f"{helpful_count / total_feedback * 100:.0f}%"
        }
    
    def save_report(self, filename='evaluation_report.json'):
        """Save evaluation report"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return filename


# Example usage for distinction-level evaluation
def run_user_testing_protocol():
    """
    Template for running distinction-level user testing
    
    Required for distinction evaluation:
    - Test with 5-10 users
    - Collect feedback
    - Show iteration based on feedback
    - Document results
    """
    
    framework = EvaluationFramework()
    
    # Test cases (would be replaced with real test cases)
    test_cases = [
        {
            'patient_data': {
                'age': 58, 'hr': 125, 'sbp': 145, 'pain': 8,
                'symptoms': ['chest pain', 'shortness of breath']
            },
            'expected_ktas': 2  # Emergent
        },
        {
            'patient_data': {
                'age': 28, 'hr': 88, 'sbp': 120, 'pain': 2,
                'symptoms': []
            },
            'expected_ktas': 5  # Non-urgent
        }
    ]
    
    # Add test cases
    for i, test in enumerate(test_cases, 1):
        framework.add_test_case(test['patient_data'], test['expected_ktas'])
    
    # Record predictions (would come from actual system)
    framework.record_prediction(test_id=1, predicted_ktas=2, confidence=0.96)
    framework.record_prediction(test_id=2, predicted_ktas=5, confidence=0.89)
    
    # Record feedback
    framework.record_feedback(
        test_id=1,
        user_feedback="System correctly identified emergency case",
        user_type='nurse'
    )
    
    # Generate and save report
    report = framework.generate_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Save to file
    framework.save_report('evaluation_report.json')
    
    return framework


if __name__ == '__main__':
    # Run example evaluation
    run_user_testing_protocol()
