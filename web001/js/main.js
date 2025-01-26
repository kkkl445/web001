// 轮播图功能
document.addEventListener('DOMContentLoaded', function() {
    const slides = document.querySelectorAll('.carousel-slide');
    const prevBtn = document.querySelector('.prev');
    const nextBtn = document.querySelector('.next');
    let currentSlide = 0;

    function showSlide(n) {
        slides.forEach(slide => slide.classList.remove('active'));
        currentSlide = (n + slides.length) % slides.length;
        slides[currentSlide].classList.add('active');
    }

    // 自动轮播
    setInterval(() => {
        showSlide(currentSlide + 1);
    }, 4000);

    // 按钮控制
    prevBtn.addEventListener('click', () => showSlide(currentSlide - 1));
    nextBtn.addEventListener('click', () => showSlide(currentSlide + 1));

    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            target.scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});

// 照片集轮播功能
document.addEventListener('DOMContentLoaded', function() {
    const gallerySlides = document.querySelectorAll('.gallery-slide');
    const galleryContainer = document.querySelector('.gallery-container');
    const prevBtn = document.querySelector('.gallery-carousel .prev');
    const nextBtn = document.querySelector('.gallery-carousel .next');
    const dotsContainer = document.querySelector('.gallery-dots');
    let currentIndex = 0;

    // 创建导航点
    gallerySlides.forEach((_, index) => {
        const dot = document.createElement('div');
        dot.classList.add('dot');
        if (index === 0) dot.classList.add('active');
        dot.addEventListener('click', () => showSlide(index));
        dotsContainer.appendChild(dot);
    });

    const dots = document.querySelectorAll('.dot');

    function showSlide(index) {
        currentIndex = index;
        galleryContainer.style.transform = `translateX(-${index * 100}%)`;
        
        // 更新导航点状态
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });
    }

    // 自动轮播
    setInterval(() => {
        currentIndex = (currentIndex + 1) % gallerySlides.length;
        showSlide(currentIndex);
    }, 5000);

    // 按钮控制
    prevBtn.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + gallerySlides.length) % gallerySlides.length;
        showSlide(currentIndex);
    });

    nextBtn.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % gallerySlides.length;
        showSlide(currentIndex);
    });

    // 触摸滑动支持
    let touchStartX = 0;
    let touchEndX = 0;

    galleryContainer.addEventListener('touchstart', e => {
        touchStartX = e.touches[0].clientX;
    });

    galleryContainer.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].clientX;
        if (touchStartX - touchEndX > 50) {
            // 向左滑动
            nextBtn.click();
        } else if (touchEndX - touchStartX > 50) {
            // 向右滑动
            prevBtn.click();
        }
    });
});

// 表单处理
document.addEventListener('DOMContentLoaded', function() {
    const messageForm = document.getElementById('messageForm');
    
    messageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 获取表单数据
        const formData = {
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            message: document.getElementById('message').value
        };
        
        // 这里可以添加发送表单数据的逻辑
        // 例如使用 fetch 发送到后端服务器
        
        // 显示提交成功消息
        alert('消息已发送！感谢您的留言。');
        
        // 重置表单
        messageForm.reset();
    });
});

// AI 对话框功能
document.addEventListener('DOMContentLoaded', function() {
    const aiChatToggle = document.getElementById('aiChatToggle');
    const aiChatClose = document.getElementById('aiChatClose');
    const aiChatBox = document.getElementById('aiChatBox');
    const aiChatInput = document.getElementById('aiChatInput');
    const aiChatSend = document.getElementById('aiChatSend');
    const aiChatMessages = document.getElementById('aiChatMessages');

    // 切换对话框显示/隐藏
    aiChatToggle.addEventListener('click', () => {
        aiChatBox.classList.add('active');
    });

    aiChatClose.addEventListener('click', () => {
        aiChatBox.classList.remove('active');
    });

    // 初始化 API 客户端
    const chatAPI = new ChatAPI(CONFIG.API_KEY);
    let isProcessing = false;
    let conversationHistory = [{
        role: 'assistant',
        content: '你好！我是 AI 助手，有什么可以帮你的吗？'
    }];

    // 发送消息
    async function sendMessage() {
        const message = aiChatInput.value.trim();
        if (message && !isProcessing) {
            try {
                isProcessing = true;
                
                // 添加用户消息到界面
                addMessage(message, true);
                aiChatInput.value = '';

                // 添加用户消息到历史记录
                conversationHistory.push({
                    role: 'user',
                    content: message
                });

                // 显示加载状态
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'ai-message';
                loadingDiv.innerHTML = `
                    <i class="fas fa-robot"></i>
                    <div class="message-content">正在思考...</div>
                `;
                aiChatMessages.appendChild(loadingDiv);
                aiChatMessages.scrollTop = aiChatMessages.scrollHeight;

                try {
                    const response = await chatAPI.sendMessage(conversationHistory);
                    loadingDiv.remove();

                    if (response && response.choices && response.choices[0]) {
                        const aiReply = response.choices[0].message.content;
                        addMessage(aiReply, false);
                        conversationHistory.push({
                            role: 'assistant',
                            content: aiReply
                        });
                    } else {
                        throw new Error('Invalid API response format');
                    }
                } catch (error) {
                    console.error('API Error:', error);
                    loadingDiv.remove();
                    addMessage('抱歉，我暂时无法回答。请稍后再试。', false);
                }

            } finally {
                isProcessing = false;
            }
        }
    }

    // 添加消息到对话框
    function addMessage(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ${isUser ? 'user-message' : ''}`;
        
        messageDiv.innerHTML = `
            ${isUser ? '' : '<i class="fas fa-robot"></i>'}
            <div class="message-content">${text}</div>
        `;
        
        aiChatMessages.appendChild(messageDiv);
        aiChatMessages.scrollTop = aiChatMessages.scrollHeight;
    }

    // 发送按钮点击事件
    aiChatSend.addEventListener('click', sendMessage);

    // 输入框回车事件
    aiChatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

// 添加导航栏滚动效果
document.addEventListener('DOMContentLoaded', function() {
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}); 