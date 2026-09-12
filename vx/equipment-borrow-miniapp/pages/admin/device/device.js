const auth = require('../../../utils/auth');
const util = require('../../../utils/util');

Page({
    data: {
        tree: [],
        expandedIds: [],
        loading: false
    },

    onLoad() {
        if (!auth.checkAdmin()) return;
        this.loadTree();
    },

    onShow() {
        this.loadTree();
    },

    async loadTree() {
        this.setData({ loading: true });
        try {
            const [catRes, devRes] = await Promise.all([
                util.callCloudFunction('device', { action: 'listAllCategories' }),
                util.callCloudFunction('device', { action: 'listAllDevices' })
            ]);
            if (catRes.code === 0 && devRes.code === 0) {
                const tree = this.buildTree(catRes.data, devRes.data.list, this.data.expandedIds);
                this.setData({ tree });
            }
        } catch (error) {
            util.showError('加载失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    buildTree(categories, devices, expandedIds) {
        const catMap = {};
        const devMap = {};
        const roots = [];
        const expanded = expandedIds || [];
        devices.forEach(d => devMap[d.categoryId] = d);
        categories.forEach(c => {
            catMap[c._id] = {
                ...c,
                device: devMap[c._id] || null,
                children: [],
                expanded: expanded.includes(c._id)
            };
        });
        categories.forEach(c => {
            if (c.parentId && catMap[c.parentId]) {
                catMap[c.parentId].children.push(catMap[c._id]);
            } else {
                roots.push(catMap[c._id]);
            }
        });
        return roots;
    },

    onToggleNode(e) {
        const id = e.currentTarget.dataset.id;
        const expandedIds = [...this.data.expandedIds];
        const idx = expandedIds.indexOf(id);
        if (idx >= 0) expandedIds.splice(idx, 1);
        else expandedIds.push(id);
        this.setData({ expandedIds });
        this.toggleNode(this.data.tree, id);
        this.setData({ tree: this.data.tree });
    },

    toggleNode(nodes, id) {
        for (let node of nodes) {
            if (node._id === id) { node.expanded = !node.expanded; return true; }
            if (node.children && this.toggleNode(node.children, id)) return true;
        }
        return false;
    },

    onAddCategory(e) {
        const parentId = e.currentTarget.dataset.parentid || '';
        const level = e.currentTarget.dataset.level || 1;
        wx.showModal({
            title: '新增',
            editable: true,
            placeholderText: '请输入名称',
            success: async (res) => {
                if (res.confirm && res.content) {
                    const result = await util.callCloudFunction('device', {
                        action: 'createCategory', name: res.content, parentId, level
                    });
                    if (result.code === 0) { util.showSuccess('创建成功'); this.loadTree(); }
                    else util.showError(result.message);
                }
            }
        });
    },

    onEditCategory(e) {
        const category = e.currentTarget.dataset.category;
        wx.showModal({
            title: '编辑', editable: true, content: category.name,
            success: async (res) => {
                if (res.confirm && res.content) {
                    const result = await util.callCloudFunction('device', {
                        action: 'updateCategory', categoryId: category._id, name: res.content
                    });
                    if (result.code === 0) { util.showSuccess('更新成功'); this.loadTree(); }
                    else util.showError(result.message);
                }
            }
        });
    },

    onDeleteCategory(e) {
        const category = e.currentTarget.dataset.category;
        if (category.children && category.children.length > 0) {
            util.showError('有子分类，无法删除'); return;
        }
        wx.showModal({
            title: '删除', content: `确定删除「${category.name}」？`,
            success: async (res) => {
                if (res.confirm) {
                    const result = await util.callCloudFunction('device', {
                        action: category.device ? 'deleteDeviceCategory' : 'deleteCategory',
                        categoryId: category._id, deviceId: category.device?._id
                    });
                    if (result.code === 0) { util.showSuccess('删除成功'); this.loadTree(); }
                    else util.showError(result.message);
                }
            }
        });
    },

    onQuantitySet(e) {
        const id = e.currentTarget.dataset.id;
        const node = this.findNode(this.data.tree, id);
        if (!node) return;

        if (node.device) {
            wx.showModal({
                title: `设置「${node.name}」数量`,
                editable: true,
                placeholderText: '请输入总数量',
                content: node.device.totalQuantity.toString(),
                success: async (res) => {
                    if (res.confirm && res.content) {
                        const qty = parseInt(res.content);
                        if (isNaN(qty) || qty < 0) { util.showError('请输入有效数量'); return; }
                        const result = await util.callCloudFunction('device', {
                            action: 'updateDeviceQuantity', deviceId: node.device._id, totalQuantity: qty
                        });
                        if (result.code === 0) { util.showSuccess('设置成功'); this.loadTree(); }
                        else util.showError(result.message);
                    }
                }
            });
        } else {
            wx.showModal({
                title: `为「${node.name}」设置数量`,
                editable: true,
                placeholderText: '请输入总数量',
                content: '0',
                success: async (res) => {
                    if (res.confirm && res.content) {
                        const qty = parseInt(res.content);
                        if (isNaN(qty) || qty < 0) { util.showError('请输入有效数量'); return; }
                        const result = await util.callCloudFunction('device', {
                            action: 'createDeviceForCategory', categoryId: node._id, quantity: qty
                        });
                        if (result.code === 0) { util.showSuccess('设置成功'); this.loadTree(); }
                        else util.showError(result.message);
                    }
                }
            });
        }
    },

    findNode(nodes, id) {
        for (let node of nodes) {
            if (node._id === id) return node;
            if (node.children) {
                const found = this.findNode(node.children, id);
                if (found) return found;
            }
        }
        return null;
    }
});
