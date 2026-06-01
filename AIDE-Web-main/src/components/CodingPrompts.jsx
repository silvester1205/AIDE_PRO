import React, { useState, useImperativeHandle } from 'react';
import { Info, Save, CheckCircle, RefreshCw } from 'lucide-react';

function CodingPrompts({ prompts, responses, onResponseChange, onRecord, onHighlightPage }, ref) {
  const [recordedIndices, setRecordedIndices] = useState(new Set());
  const [showSource, setShowSource] = useState(new Set());

  const handleRecord = (index) => {
    onRecord(index, responses[index].response);
    setRecordedIndices(prev => new Set([...prev, index]));
  };

  const handleReRecord = (index) => {
    setRecordedIndices(prev => {
      const newSet = new Set(prev);
      newSet.delete(index);
      return newSet;
    });
  };

  const toggleSource = (index) => {
    const isShowing = showSource.has(index);
    setShowSource(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) newSet.delete(index);
      else newSet.add(index);
      return newSet;
    });

    // When opening source, try to highlight the page in PDF viewer
    if (!isShowing && onHighlightPage && responses[index]?.page) {
      const pageStr = String(responses[index].page);
      // Skip if page is N/A or Not found
      if (pageStr === 'N/A' || pageStr === 'Not found' || pageStr === 'Not specified') {
        console.log('Page is not available:', pageStr);
        return;
      }
      // Extract page number from strings like "Page 2", "Page: 2", "2", etc.
      const pageMatch = pageStr.match(/\d+/);
      if (pageMatch) {
        const pageNum = parseInt(pageMatch[0], 10);
        if (pageNum > 0) {
          console.log('Highlighting page:', pageNum);
          onHighlightPage(pageNum);
        }
      } else {
        console.log('Could not extract page number from:', pageStr);
      }
    }
  };

  const resetRecordedIndices = () => {
    setRecordedIndices(new Set());
  };

  // Expose the resetRecordedIndices method to the parent component
  useImperativeHandle(ref, () => ({
    resetRecordedIndices
  }));

  return (
    <div style={{ maxHeight: '75vh', overflowY: 'auto', paddingRight: '0.5rem' }}>
      {prompts.map((prompt, index) => {
        const isRecorded = recordedIndices.has(index);
        return (
          <div key={index} style={{
            marginBottom: '1.5rem',
            paddingBottom: '1.5rem',
            paddingLeft: '0.75rem',
            borderBottom: '1px solid #dee2e6',
            borderLeft: isRecorded ? '4px solid #28a745' : '4px solid transparent',
            backgroundColor: isRecorded ? '#f0fff4' : 'transparent',
            borderRadius: '0 0.375rem 0.375rem 0',
            transition: 'all 0.3s ease',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <h4 style={{
                fontSize: '1rem',
                fontWeight: 600,
                margin: 0,
                color: isRecorded ? '#1a6b30' : '#2c3e50',
                flex: 1,
              }}>
                Prompt {index + 1}: {prompt}
              </h4>
              {isRecorded && (
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: '#28a745',
                  backgroundColor: '#d4edda',
                  padding: '0.2rem 0.6rem',
                  borderRadius: '999px',
                  whiteSpace: 'nowrap',
                }}>
                  <CheckCircle size={12} />
                  Recorded
                </span>
              )}
            </div>

            <textarea
              className="form-textarea"
              value={responses[index]?.response || ''}
              onChange={(e) => onResponseChange(index, e.target.value)}
              placeholder="Response will appear here after analysis..."
              style={{
                minHeight: '100px',
                borderColor: isRecorded ? '#28a745' : undefined,
                backgroundColor: isRecorded ? '#f8fff9' : undefined,
              }}
            />

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <button
                className="btn btn-info"
                onClick={() => toggleSource(index)}
                disabled={!responses[index]?.source}
              >
                <Info size={16} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
                Source
              </button>

              {isRecorded ? (
                <>
                  <button
                    className="btn btn-success"
                    disabled
                    style={{ opacity: 1, cursor: 'default' }}
                  >
                    <CheckCircle size={16} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
                    Recorded ✓
                  </button>
                  <button
                    className="btn"
                    onClick={() => handleReRecord(index)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#6c757d',
                      fontSize: '0.8rem',
                      padding: '0.25rem 0.5rem',
                      textDecoration: 'underline',
                      cursor: 'pointer',
                    }}
                  >
                    <RefreshCw size={12} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
                    Re-record
                  </button>
                </>
              ) : (
                <button
                  className="btn btn-primary"
                  onClick={() => handleRecord(index)}
                  disabled={!responses[index]?.response}
                >
                  <Save size={16} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
                  Record
                </button>
              )}
            </div>

            {showSource.has(index) && responses[index]?.source && (
              <div style={{
                marginTop: '0.75rem',
                padding: '1rem',
                backgroundColor: '#fffbeb',
                border: '2px solid #f59e0b',
                borderRadius: '0.5rem',
                fontSize: '0.9rem',
                boxShadow: '0 2px 8px rgba(245, 158, 11, 0.15)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    fontWeight: 700,
                    color: '#92400e',
                    fontSize: '0.95rem'
                  }}>
                    📄 Source Text
                  </span>
                  <button
                    className="btn btn-secondary"
                    onClick={() => navigator.clipboard.writeText(responses[index].source)}
                    style={{ padding: '0.25rem 0.75rem', fontSize: '0.8rem' }}
                  >
                    Copy
                  </button>
                </div>

                {/* 位置信息：章节 + 页码 + 页面位置 */}
                <div style={{
                  marginBottom: '0.75rem',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.5rem'
                }}>
                  {responses[index].section && responses[index].section !== 'Not specified' && (
                    <span style={{
                      padding: '0.35rem 0.75rem',
                      backgroundColor: '#fce7f3',
                      color: '#be185d',
                      borderRadius: '999px',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      border: '1px solid #fbcfe8'
                    }}>
                      📑 {responses[index].section}
                    </span>
                  )}
                  {responses[index].page && responses[index].page !== 'N/A' && (
                    <span style={{
                      padding: '0.35rem 0.75rem',
                      backgroundColor: '#dbeafe',
                      color: '#1e40af',
                      borderRadius: '999px',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      border: '1px solid #bfdbfe'
                    }}>
                      📍 {responses[index].page}
                    </span>
                  )}
                  {responses[index].location && responses[index].location !== 'Not specified' && (
                    <span style={{
                      padding: '0.35rem 0.75rem',
                      backgroundColor: '#dcfce7',
                      color: '#166534',
                      borderRadius: '999px',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      border: '1px solid #bbf7d0'
                    }}>
                      🎯 {responses[index].location}
                    </span>
                  )}
                </div>

                <div style={{
                  padding: '0.75rem',
                  backgroundColor: '#fef3c7',
                  borderRadius: '0.375rem',
                  borderLeft: '4px solid #f59e0b'
                }}>
                  <p style={{
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    color: '#78350f',
                    lineHeight: '1.7',
                    fontWeight: 500
                  }}>
                    {responses[index].source}
                  </p>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default React.forwardRef(CodingPrompts);