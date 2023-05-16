const searchInput = document.querySelector('#search');
const rows = document.querySelectorAll('table tbody tr');
const sortBtns = document.querySelectorAll(".sort-btn");


searchInput.addEventListener('keyup', () => {
	const searchTerm = searchInput.value.toLowerCase().trim();

	rows.forEach(row => {
		const title = row.querySelector('td:nth-child(2)').textContent.toLowerCase();
		const channel = row.querySelector('td:nth-child(3)').textContent.toLowerCase();

		if (title.indexOf(searchTerm) > -1 || channel.indexOf(searchTerm) > -1) {
			row.style.display = '';
		} else {
			row.style.display = 'none';
		}
	});

	  

})

