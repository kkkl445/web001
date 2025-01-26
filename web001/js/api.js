class ChatAPI {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseURL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions';
    }

    // 生成 JWT token
    async generateToken() {
        try {
            const [id, secret] = this.apiKey.split('.');
            const now = Date.now();
            
            const payload = {
                api_key: id,
                exp: now + 3600 * 1000,
                timestamp: now
            };

            const header = {
                alg: 'HS256',
                sign_type: 'SIGN'
            };

            // 使用 HMAC SHA256 算法手动生成签名
            const base64Header = btoa(JSON.stringify(header));
            const base64Payload = btoa(JSON.stringify(payload));
            
            const signContent = `${base64Header}.${base64Payload}`;
            const signature = await this.hmacSha256(signContent, secret);
            
            return `${signContent}.${signature}`;
        } catch (error) {
            console.error('Error generating token:', error);
            throw new Error('Token generation failed: ' + error.message);
        }
    }

    // HMAC SHA256 签名
    async hmacSha256(message, secret) {
        const encoder = new TextEncoder();
        const keyData = encoder.encode(secret);
        const messageData = encoder.encode(message);

        const cryptoKey = await crypto.subtle.importKey(
            'raw',
            keyData,
            { name: 'HMAC', hash: 'SHA-256' },
            false,
            ['sign']
        );

        const signature = await crypto.subtle.sign(
            'HMAC',
            cryptoKey,
            messageData
        );

        return btoa(String.fromCharCode(...new Uint8Array(signature)));
    }

    // 发送消息到 API
    async sendMessage(messages) {
        try {
            const token = await this.generateToken();
            console.log('Sending request to API...');
            
            const requestBody = {
                model: 'glm-4',
                messages: messages,
                stream: false
            };
            
            console.log('Request body:', requestBody);

            const response = await fetch(this.baseURL, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('API error response:', errorText);
                throw new Error(`API request failed with status ${response.status}`);
            }

            const data = await response.json();
            console.log('API response:', data);
            return data;
        } catch (error) {
            console.error('Error calling API:', error);
            throw error;
        }
    }
} 