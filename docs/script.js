Vue.component('sample-state-space', {
    props: ['id'],
    template: '#sample-state-space',
    data() {
      return {
        current_step: 0,
      }
    },
    methods: {
      
    }
});

var v = new Vue({
    el: '#content',
    data: {
      showAllSamples: false,
    },
    methods: {
        showMore: function() {
            this.showAllSamples = true;
        },
    }
});
