import time
import streamlit as st
import psutil
import pandas as pd

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics['cpu_start'] = psutil.cpu_percent()
        self.metrics['memory_start'] = psutil.virtual_memory().percent
    
    def stop_monitoring(self):
        """Stop monitoring and calculate metrics"""
        if self.start_time:
            self.metrics['execution_time'] = time.time() - self.start_time
            self.metrics['cpu_end'] = psutil.cpu_percent()
            self.metrics['memory_end'] = psutil.virtual_memory().percent
            return self.metrics
        return None
    
    def display_metrics(self):
        """Display performance metrics in Streamlit"""
        if self.metrics:
            with st.expander("⚡ Performance Metrics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Execution Time", f"{self.metrics.get('execution_time', 0):.2f}s")
                
                with col2:
                    cpu_usage = self.metrics.get('cpu_end', 0)
                    st.metric("CPU Usage", f"{cpu_usage:.1f}%")
                
                with col3:
                    memory_usage = self.metrics.get('memory_end', 0)
                    st.metric("Memory Usage", f"{memory_usage:.1f}%")