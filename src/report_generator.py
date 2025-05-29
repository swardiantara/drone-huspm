import os
import contextlib
from datetime import datetime
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_loader import LogRecord

class ReportGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir: str = None

    def parse_datetime(self, date_str, time_str):
        """Parse date and time strings into datetime object"""
        # Handle AM/PM format
        datetime_str = f"{date_str} {time_str}"
        try:
            # Try parsing with microseconds
            return datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S.%f %p")
        except ValueError:
            # Fallback without microseconds
            return datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S %p")

    def create_timeline_chart(self, records_list: List[LogRecord], problems: dict, output_dir: str):
        """Create a Gantt-style timeline chart showing drone problems"""
        self.output_dir = output_dir
        # Load data
        # records_list, problems = load_json_files(records_file, problems_file)
        
        # Process all records and collect problem events
        all_problem_events = []
        
        for record in records_list:
            # Parse datetime for each record
            timestamp = self.parse_datetime(record['date'], record['time'])
            
            # Find matching problem events in this record
            for i, event_id in enumerate(record['eventIds']):
                if event_id in problems:
                    all_problem_events.append({
                        'timestamp': timestamp,
                        'event_id': event_id,
                        'sentence': problems[event_id],
                        'anomaly_level': record['anomalies'][i]
                    })
        
        if not all_problem_events:
            print("No problem events found in any records.")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Get unique problem sentences for y-axis
        unique_sentences = list(set([event['event_id'] for event in all_problem_events]))
        y_positions = {sentence: i for i, sentence in enumerate(unique_sentences)}
        
        # Color mapping for anomaly levels
        color_map = {
            'low': '#FFA500',      # Orange
            'medium': '#FF4500',   # Red-Orange  
            'high': 'purple'      # Crimson
        }
        
        # Plot each problem event
        for event in all_problem_events:
            y_pos = y_positions[event['event_id']]
            color = color_map.get(event['anomaly_level'], '#FF0000')  # Default to red
            
            ax.scatter(event['timestamp'], y_pos, 
                    color=color, s=7, alpha=0.8, zorder=5)
        
        # Customize the plot
        ax.set_yticks(range(len(unique_sentences)))
        ax.set_yticklabels(unique_sentences, fontsize=10)
        ax.set_xlabel('Timeline', fontsize=11)
        ax.set_ylabel('Problem Events', fontsize=11)
        # ax.set_title('Drone Problem Timeline - Gantt Style Visualization', fontsize=14, fontweight='bold')
        
        # Format x-axis for time display
        if len(all_problem_events) > 0:
            # Get time range to determine appropriate formatting
            timestamps = [event['timestamp'] for event in all_problem_events]
            time_span = max(timestamps) - min(timestamps)
            
            if time_span.total_seconds() < 3600:  # Less than 1 hour
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            else:  # More than 1 hour
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend for anomaly levels (create manually to avoid duplicates)
        legend_elements = []
        anomaly_levels_present = set([event['anomaly_level'] for event in all_problem_events])
        for level in ['low', 'medium', 'high']:
            if level in anomaly_levels_present:
                legend_elements.append(plt.scatter([], [], color=color_map[level], s=10, label=f"{level}"))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.45, 1.17), ncol=3)
        
        # Adjust layout
        plt.tight_layout()
        # Save the chart into a .PDF file
        plt.savefig(os.path.join(output_dir, f'{self.config['filename']}_timeline.pdf'))
        # Show timeline plot
        plt.show()
        
        # Generate and display summary statistics
        stats = self.generate_summary_statistics(all_problem_events, records_list, problems)
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            with contextlib.redirect_stdout(f):
                self.print_investigation_summary(stats)
        
        # Create summary dashboard
        self.create_summary_dashboard(stats)
        
        return stats  # Return stats for further analysis if needed

    def calculate_flight_risk_score(self, all_problem_events, total_records):
        """Calculate a risk score for the flight based on problem frequency and severity"""
        if not all_problem_events:
            return 0, "LOW"
        
        # Weight factors for different anomaly levels
        weights = {'low': 1, 'medium': 2, 'high': 3}
        
        # Calculate weighted problem score
        weighted_score = sum(weights.get(event['anomaly_level'], 1) for event in all_problem_events)
        
        # Normalize by total records to get problems per record
        problems_per_record = len(all_problem_events) / max(total_records, 1)
        
        # Combined risk score (0-100 scale)
        risk_score = min(100, (weighted_score * problems_per_record * 10))
        
        # Risk categories
        if risk_score < 10:
            risk_level = "LOW"
        elif risk_score < 30:
            risk_level = "MEDIUM"
        elif risk_score < 60:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return round(risk_score, 1), risk_level

    def generate_summary_statistics(self, all_problem_events, records_list, problems):
        """Generate comprehensive summary statistics for investigators"""
        
        if not all_problem_events:
            return {
                'risk_score': 0,
                'risk_level': 'LOW',
                'total_problems': 0,
                'recommendation': 'No problems detected - Low priority for investigation'
            }
        
        # Basic counts
        total_records = len(records_list)
        total_problems = len(all_problem_events)
        
        # Risk assessment
        risk_score, risk_level = self.calculate_flight_risk_score(all_problem_events, total_records)
        
        # Anomaly distribution
        anomaly_counts = Counter([event['anomaly_level'] for event in all_problem_events])
        
        # Problem type distribution
        problem_type_counts = Counter([event['event_id'] for event in all_problem_events])
        
        # Temporal analysis
        timestamps = [event['timestamp'] for event in all_problem_events]
        flight_duration = max(timestamps) - min(timestamps)
        
        # Problem clustering (problems per minute)
        problem_rate = total_problems / max(flight_duration.total_seconds() / 60, 1)
        
        # Critical event detection
        critical_events = [event for event in all_problem_events if event['anomaly_level'] == 'high']
        
        # Investigation recommendation
        recommendations = []
        if risk_level == "CRITICAL":
            recommendations.append("IMMEDIATE INVESTIGATION REQUIRED")
        elif risk_level == "HIGH":
            recommendations.append("High priority - Investigate within 24 hours")
        elif risk_level == "MEDIUM":
            recommendations.append("Medium priority - Investigate within 1 week")
        else:
            recommendations.append("Low priority - Routine review sufficient")
        
        if len(critical_events) > 0:
            recommendations.append(f"⚠️  {len(critical_events)} HIGH severity events detected")
        
        if problem_rate > 2:  # More than 2 problems per minute
            recommendations.append("⚠️  High problem frequency detected")
        
        if 'RC signal lost' in [event['sentence'] for event in all_problem_events]:
            recommendations.append("⚠️  Communication loss detected - Safety critical")
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'total_problems': total_problems,
            'total_records': total_records,
            'problem_density': round(total_problems / total_records, 2),
            'flight_duration_minutes': round(flight_duration.total_seconds() / 60, 1),
            'problem_rate_per_minute': round(problem_rate, 2),
            'anomaly_distribution': dict(anomaly_counts),
            'top_problems': dict(problem_type_counts.most_common(3)),
            'critical_events_count': len(critical_events),
            'recommendations': recommendations,
            'timeline_start': min(timestamps),
            'timeline_end': max(timestamps)
        }

    def print_investigation_summary(self, stats):
        """Print a formatted summary for investigators"""
        
        print("\n" + "="*80)
        print("🔍 FLIGHT LOG INVESTIGATION SUMMARY")
        print("="*80)
        
        # Risk Assessment Header
        risk_color = {
            'LOW': '🟢',
            'MEDIUM': '🟡', 
            'HIGH': '🟠',
            'CRITICAL': '🔴'
        }
        
        print(f"\n📊 RISK ASSESSMENT:")
        print(f"   Risk Level: {risk_color.get(stats['risk_level'], '⚪')} {stats['risk_level']}")
        print(f"   Risk Score: {stats['risk_score']}/100")
        
        # Key Metrics
        print(f"\n📈 KEY METRICS:")
        print(f"   Total Problem Events: {stats['total_problems']}")
        print(f"   Flight Duration: {stats['flight_duration_minutes']} minutes")
        print(f"   Problem Density: {stats['problem_density']} problems/record")
        print(f"   Problem Rate: {stats['problem_rate_per_minute']} problems/minute")
        
        # Severity Breakdown
        print(f"\n⚠️  SEVERITY BREAKDOWN:")
        for level in ['high', 'medium', 'low']:
            count = stats['anomaly_distribution'].get(level, 0)
            if count > 0:
                print(f"   {level.upper()}: {count} events")
        
        # Top Problems
        print(f"\n🎯 TOP PROBLEM TYPES:")
        for i, (problem, count) in enumerate(stats['top_problems'].items(), 1):
            print(f"   {i}. {problem} ({count} times)")
        
        # Timeline
        print(f"\n⏰ TIMELINE:")
        print(f"   Start: {stats['timeline_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End: {stats['timeline_end'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Recommendations
        print(f"\n💡 INVESTIGATION RECOMMENDATIONS:")
        for rec in stats['recommendations']:
            print(f"   • {rec}")
        
        print("\n" + "="*80)

    def create_summary_dashboard(self, stats):
        """Create visual summary dashboard"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flight Log Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Risk Score Gauge
        risk_score = stats['risk_score']
        colors = ['green' if risk_score < 10 else 'orange' if risk_score < 30 else 'red' if risk_score < 60 else 'darkred']
        ax1.pie([risk_score, 100-risk_score], labels=['Risk', 'Safe'], colors=[colors[0], 'lightgray'],
                startangle=90, counterclock=False)
        ax1.set_title(f'Risk Score: {risk_score}/100\n({stats["risk_level"]})', fontweight='bold')
        
        # 2. Anomaly Distribution
        if stats['anomaly_distribution']:
            levels = list(stats['anomaly_distribution'].keys())
            counts = list(stats['anomaly_distribution'].values())
            colors_map = {'low': 'orange', 'medium': 'red', 'high': 'darkred'}
            bar_colors = [colors_map.get(level, 'gray') for level in levels]
            ax2.bar(levels, counts, color=bar_colors, alpha=0.7)
            ax2.set_title('Problem Severity Distribution')
            ax2.set_ylabel('Count')
        else:
            ax2.text(0.5, 0.5, 'No Problems\nDetected', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Problem Severity Distribution')
        
        # 3. Top Problems
        if stats['top_problems']:
            problems = list(stats['top_problems'].keys())
            counts = list(stats['top_problems'].values())
            # Truncate long problem names
            problems = [p[:30] + '...' if len(p) > 30 else p for p in problems]
            ax3.barh(problems, counts, color='skyblue', alpha=0.7)
            ax3.set_title('Most Frequent Problems')
            ax3.set_xlabel('Frequency')
        else:
            ax3.text(0.5, 0.5, 'No Problems\nDetected', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Most Frequent Problems')
        
        # 4. Key Metrics Text
        ax4.axis('off')
        metrics_text = f"""
            KEY METRICS:

            Total Problems: {stats['total_problems']}
            Flight Duration: {stats['flight_duration_minutes']} min
            Problem Rate: {stats['problem_rate_per_minute']}/min
            Problem Density: {stats['problem_density']}/record

            INVESTIGATION PRIORITY:
            {stats['risk_level']}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.config['filename']}_dashboard.pdf'))
        plt.show()