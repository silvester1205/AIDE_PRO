/**
 * Call LLM API with structured JSON request and response
 * @param {string} endpoint - API endpoint URL
 * @param {string} apiKey - API key
 * @param {string} model - Model name to use
 * @param {Array<string>} prompts - Array of prompts
 * @param {Object} content - Content object with type and data
 * @param {number|null} contextWindow - Optional context window size
 * @returns {Promise<Object>} - Structured response
 */
export async function callLLMAPI(endpoint, apiKey, model, prompts, content, contextWindow = null) {
  try {
    // Detect if using local LMStudio/lm.cpp endpoint (not OpenRouter on local proxy)
    const isLocalLMStudio = /localhost|127\.0\.0\.1|192\.168\.|10\.|:\d{4,5}\/v1\/chat\/completions/i.test(endpoint)
      && !endpoint.includes('openrouter')
      && !endpoint.includes('proxy');

    const responseSchema = {
      type: 'object',
      additionalProperties: false,
      properties: {
        responses: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              prompt: { type: 'string' },
              response: { type: 'string' },
              source: { type: 'string' },
              page: { type: 'string' },
              section: { type: 'string' },
              location: { type: 'string' }
            },
            required: ['prompt', 'response', 'source', 'page']
          }
        }
      },
      required: ['responses']
    };

    const extractionTool = {
      type: 'function',
      function: {
        name: 'submit_extraction',
        description: 'Submit extracted answers for all prompts with supporting source text and page.',
        parameters: responseSchema
      }
    };

    const systemMessage = isLocalLMStudio
      ? `You are a data extraction assistant. Extract information from documents and return ONLY valid JSON.

RULES:
1. Return ONLY a JSON object, no markdown, no explanations, no code blocks
2. Use this exact structure: {"responses":[{"prompt":"...","response":"...","source":"...","page":"...","section":"...","location":"..."}]}
3. For each prompt, provide: response (the answer), source (exact text from document), page (number), section (heading), location (position on page)
4. If info not found, use "Not found" for response/source/section/location
5. Do NOT wrap in \`\`\`json or any other formatting`
      : `You are a data extraction assistant for systematic reviews and meta-analysis.
Extract the requested information from the provided document.

For each response, you MUST provide:
1. response: The extracted answer
2. source: The EXACT original text snippet from the document that supports your answer (quote the relevant sentence or paragraph, not just "Document Text")
3. page: The specific page number where this information was found
4. section: The section/heading where this information is located (e.g., "Results", "Methods", "Table 1", "Figure 2", "Introduction", etc.)
5. location: Brief description of the location on the page (e.g., "first paragraph", "second column", "bottom of page", "upper right section", "table row 3", etc.)

Be accurate and concise. If information is not found, use "Not found" for response, source, section and location fields.`;

    // Build the user message with prompts
    let userMessage = '';

    if (content.type === 'text') {
      userMessage = `Document Text:\n${content.data}\n\n`;
    } else if (content.type === 'pdf') {
      userMessage = `[PDF Document: ${content.fileName}]\n\n`;
    }

    userMessage += `Prompts to answer:\n`;
    prompts.forEach((prompt, index) => {
      userMessage += `${index + 1}. ${prompt}\n`;
    });

    // Build example response based on actual prompts
    const exampleResponses = prompts.slice(0, 2).map((p, idx) => ({
      prompt: p,
      response: "extracted answer or Not found",
      source: "exact text from document or Not found",
      page: String(idx + 1),
      section: "section name",
      location: "location on page"
    }));

    // If only one prompt, add a placeholder second example
    if (exampleResponses.length === 1) {
      exampleResponses.push({
        prompt: "second prompt example",
        response: "extracted answer or Not found",
        source: "exact text from document or Not found",
        page: "2",
        section: "section name",
        location: "location on page"
      });
    }

    userMessage += `\nExtract information for each prompt and return ONLY a JSON object.\n\nREQUIRED FORMAT (copy this structure, replace with your answers):\n${JSON.stringify({ responses: exampleResponses })}\n\nCRITICAL RULES:\n1. Return ONLY the JSON - no markdown \`\`\`, no explanations, no extra text\n2. "responses" must be an array with ${prompts.length} item(s)\n3. Every item MUST have: prompt, response, source, page, section, location\n4. If not found, use "Not found" as the value\n5. Do not include any text before or after the JSON`;

    // Build the request
    const messages = [
      { role: 'system', content: systemMessage },
      { role: 'user', content: userMessage }
    ];

    // For PDF mode with vision-capable models, include the PDF
    if (content.type === 'pdf') {
      // Try to use vision API if available (e.g., GPT-4 Vision, Claude with images)
      // For now, we'll convert PDF to text as a fallback
      // You can extend this to handle base64 images for vision models
    }

    const structuredRequestBody = {
      model: model, // Use the model passed in directly
      messages: messages,
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'extraction_result',
          strict: true,
          schema: responseSchema
        }
      },
      tools: [extractionTool]
    };

    if (contextWindow) {
      structuredRequestBody.max_tokens = contextWindow;
    }

    const fallbackRequestBody = {
      model,
      messages,
      response_format: { type: 'json_object' }
    };

    if (contextWindow) {
      fallbackRequestBody.max_tokens = contextWindow;
    }

    // LMStudio-compatible minimal request (no response_format, no tools)
    const simpleRequestBody = {
      model,
      messages,
      temperature: 0.1,
      max_tokens: contextWindow || -1
    };
    // Remove max_tokens if not provided (LMStudio handles -1 as unlimited)
    if (!contextWindow) {
      delete simpleRequestBody.max_tokens;
    }

    const sendRequest = async (body) => {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(body)
      });

      const payload = await response.json().catch(() => ({}));
      return { response, payload };
    };

    let response, data;

    // For local LMStudio, try simple request first
    if (isLocalLMStudio) {
      console.log('Detected local endpoint, trying simple request first...');
      const simpleResult = await sendRequest(simpleRequestBody);
      response = simpleResult.response;
      data = simpleResult.payload;

      // If simple request works, we'll parse the JSON from content manually
      if (response.ok) {
        console.log('Simple request succeeded for local endpoint');
      } else {
        console.warn('Simple request failed, trying fallback...');
        const fallbackResult = await sendRequest(fallbackRequestBody);
        response = fallbackResult.response;
        data = fallbackResult.payload;
      }
    } else {
      // Cloud endpoints: try structured first, then fallback
      let result = await sendRequest(structuredRequestBody);
      response = result.response;
      data = result.payload;

      if (!response.ok) {
        const message = data?.error?.message || '';
        const shouldFallback =
          response.status === 400 &&
          /(json_schema|response_format|tools?|tool_choice|unsupported|unknown)/i.test(message);

        if (shouldFallback) {
          console.warn('Structured output request unsupported. Retrying with json_object mode.');
          const fallbackResult = await sendRequest(fallbackRequestBody);
          response = fallbackResult.response;
          data = fallbackResult.payload;
        }
      }
    }

    if (!response.ok) {
      throw new Error(
        data?.error?.message ||
        `API request failed with status ${response.status}`
      );
    }

    // console.log('Full API response:', data);

    // Extract structured output from either message.content or tool_calls.
    const message = data.choices?.[0]?.message || {};
    const toolCalls = message.tool_calls || [];
    // console.log('Raw LLM message content:', message.content);
    // console.log('Raw LLM tool calls:', toolCalls);

    // Handle LMStudio simple mode: if content is JSON string but not parsed
    let contentFromSimpleMode = null;
    if (isLocalLMStudio && message.content && typeof message.content === 'string') {
      // Try to find JSON in the content (LLM might wrap it in markdown or add text)
      const jsonMatch = message.content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          const parsed = JSON.parse(jsonMatch[0]);
          if (parsed.responses || Array.isArray(parsed)) {
            contentFromSimpleMode = parsed;
            console.log('Parsed JSON from simple mode content');
          }
        } catch (e) {
          // Not valid JSON, will be handled below
        }
      }
    }

    const contentToText = (value) => {
      if (typeof value === 'string') {
        return value;
      }

      if (Array.isArray(value)) {
        return value
          .map((part) => {
            if (typeof part === 'string') {
              return part;
            }

            // OpenAI/Anthropic compatibility wrappers may return content blocks.
            if (part?.type === 'text' && typeof part?.text === 'string') {
              return part.text;
            }

            return '';
          })
          .join('')
          .trim();
      }

      return '';
    };

    const tryParseJson = (rawText) => {
      if (!rawText || typeof rawText !== 'string') {
        return null;
      }

      const cleanedText = rawText
        .replace(/\`\`\`json\s*/gi, '')
        .replace(/\`\`\`\s*/g, '')
        .trim();

      try {
        return JSON.parse(cleanedText);
      } catch {
        return null;
      }
    };

    let parsedResponse = null;

    // 0) LMStudio simple mode: use parsed JSON from content
    if (contentFromSimpleMode) {
      parsedResponse = contentFromSimpleMode;
    }

    // 1) Preferred: JSON in message.content (from response_format=json_schema or json_object)
    if (!parsedResponse) {
      const contentText = contentToText(message.content);
      parsedResponse = tryParseJson(contentText);
    }

    // 2) Fallback: JSON function arguments in tool_calls (common with Claude adapters)
    if (!parsedResponse && toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        const argsText = toolCall?.function?.arguments;
        const parsedArgs = tryParseJson(argsText);
        if (parsedArgs) {
          parsedResponse = parsedArgs;
          break;
        }
      }
    }

    if (!parsedResponse) {
      console.error('Failed to parse response. Content:', message.content, 'Tool calls:', toolCalls);
      console.error('Raw response data:', JSON.stringify(data, null, 2));

      // Try one more time with more aggressive extraction for local models
      if (isLocalLMStudio && message.content) {
        const rawContent = String(message.content);
        // Try to find anything that looks like a JSON array or object with responses
        const possibleJson = rawContent.match(/\{[\s\S]*"responses"[\s\S]*\}/);
        if (possibleJson) {
          try {
            const extracted = JSON.parse(possibleJson[0]);
            if (extracted.responses || Array.isArray(extracted)) {
              parsedResponse = extracted;
              console.log('Recovered JSON via aggressive extraction');
            }
          } catch (e) {
            // Failed to recover
          }
        }
      }

      if (!parsedResponse) {
        throw new Error('LLM did not return parseable structured output. Please try again.');
      }
    }

    // Normalize common provider variants before strict validation.
    if (Array.isArray(parsedResponse)) {
      parsedResponse = { responses: parsedResponse };
    }

    const responsesValue = parsedResponse?.responses;
    if (typeof responsesValue === 'string') {
      const parsedResponses = tryParseJson(responsesValue);

      if (Array.isArray(parsedResponses)) {
        parsedResponse.responses = parsedResponses;
      } else if (parsedResponses && Array.isArray(parsedResponses.responses)) {
        parsedResponse.responses = parsedResponses.responses;
      }
    }

    // Validate the response structure
    if (!parsedResponse.responses || !Array.isArray(parsedResponse.responses)) {
      throw new Error('Invalid response structure from LLM');
    }

    // Ensure all prompts have responses and required shape for downstream UI.
    parsedResponse.responses = prompts.map((prompt, index) => {
      const existing = parsedResponse.responses[index] || {};
      return {
        prompt,
        response: existing.response || 'Not found',
        source: existing.source || 'Not found',
        page: existing.page || 'N/A',
        section: existing.section || 'Not specified',
        location: existing.location || 'Not specified'
      };
    });

    return parsedResponse;

  } catch (error) {
    console.error('API call error:', error);
    throw error;
  }
}

/**
 * Test API connection
 * @param {string} endpoint - API endpoint URL
 * @param {string} apiKey - API key
 * @param {string} model - Model name to use
 * @returns {Promise<boolean>} - True if connection successful
 */
export async function testAPIConnection(endpoint, apiKey, model) {
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: model,
        messages: [{ role: 'user', content: 'test' }],
        max_tokens: 5
      })
    });

    return response.ok;
  } catch (error) {
    console.error('API test failed:', error);
    return false;
  }
}